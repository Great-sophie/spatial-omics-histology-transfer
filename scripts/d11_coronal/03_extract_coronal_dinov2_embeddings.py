import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =========================================================
# Configuration
# =========================================================

ROOT = Path(
    "/media/sophie/Data2T/Biomedical AI  skill/"
    "D11_mouse_brain_coronal_validation"
)

OUT = ROOT / "outputs"

MANIFEST_PATH = (
    OUT
    / "coronal_multiscale_patch_manifest.csv"
)

LOW_OUT = (
    OUT
    / "dinov2_vits14_coronal_lowres_embeddings.npy"
)

FULL_OUT = (
    OUT
    / "dinov2_vits14_coronal_fullres_embeddings.npy"
)

EMBED_MANIFEST_OUT = (
    OUT
    / "dinov2_vits14_coronal_embedding_manifest.csv"
)

SUMMARY_OUT = (
    OUT
    / "dinov2_vits14_coronal_embedding_summary.csv"
)


MODEL_NAME = "dinov2_vits14"

BATCH_SIZE = 16
NUM_WORKERS = 4

EXPECTED_DIM = 384


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )


# =========================================================
# Manifest
# =========================================================

print(
    "\nLoading patch manifest..."
)


manifest = pd.read_csv(
    MANIFEST_PATH
)


manifest[
    "barcode"
] = (
    manifest[
        "barcode"
    ]
    .astype(str)
)


required = [
    "barcode",
    "lowres_patch_path",
    "fullres_patch_path",
]


for col in required:

    if col not in manifest.columns:

        raise KeyError(
            f"Missing column: {col}"
        )


print(
    "Spots:",
    len(manifest)
)


# =========================================================
# Validate patch files
# =========================================================

missing_low = 0
missing_full = 0


for _, row in manifest.iterrows():

    if not Path(
        row[
            "lowres_patch_path"
        ]
    ).exists():

        missing_low += 1


    if not Path(
        row[
            "fullres_patch_path"
        ]
    ).exists():

        missing_full += 1


print(
    "Missing low:",
    missing_low
)

print(
    "Missing full:",
    missing_full
)


if (
    missing_low > 0
    or
    missing_full > 0
):

    raise FileNotFoundError(
        "Patch files incomplete."
    )


# =========================================================
# Image preprocessing
#
# Same preprocessing as D9/D10.
# =========================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224),
            interpolation=
                transforms.InterpolationMode.BICUBIC,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],

            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# =========================================================
# Dataset
# =========================================================

class PairedDataset(Dataset):

    def __init__(
        self,
        df,
        transform,
    ):

        self.df = (
            df
            .reset_index(
                drop=True
            )
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.df
        )


    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[
            idx
        ]


        low = Image.open(
            row[
                "lowres_patch_path"
            ]
        ).convert("RGB")


        full = Image.open(
            row[
                "fullres_patch_path"
            ]
        ).convert("RGB")


        low = self.transform(
            low
        )

        full = self.transform(
            full
        )


        return (
            low,
            full,
            idx,
        )


dataset = PairedDataset(
    manifest,
    transform,
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(
        NUM_WORKERS > 0
    ),
)


# =========================================================
# Load DINOv2
# =========================================================

print(
    "\nLoading DINOv2 ViT-S/14..."
)


model = torch.hub.load(
    "facebookresearch/dinov2",
    MODEL_NAME,
)


model = model.to(
    device
)


model.eval()


for p in model.parameters():

    p.requires_grad = False


print(
    "Model loaded."
)


# =========================================================
# CLS embedding
# =========================================================

def get_embedding(
    images,
):

    features = model.forward_features(
        images
    )


    if isinstance(
        features,
        dict
    ):

        if (
            "x_norm_clstoken"
            not in features
        ):

            raise KeyError(
                "x_norm_clstoken missing."
            )


        return features[
            "x_norm_clstoken"
        ]


    return features


# =========================================================
# Extract
# =========================================================

low_embeddings = []
full_embeddings = []
indices = []


start = time.time()


print(
    "\nExtracting embeddings..."
)


with torch.inference_mode():

    for batch_idx, (
        low,
        full,
        idx
    ) in enumerate(loader):


        low = low.to(
            device,
            non_blocking=True
        )


        full = full.to(
            device,
            non_blocking=True
        )


        images = torch.cat(
            [
                low,
                full
            ],
            dim=0
        )


        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                emb = get_embedding(
                    images
                )

        else:

            emb = get_embedding(
                images
            )


        emb = (
            emb
            .float()
            .cpu()
            .numpy()
        )


        n = low.shape[
            0
        ]


        low_embeddings.append(
            emb[
                :n
            ]
        )


        full_embeddings.append(
            emb[
                n:
            ]
        )


        indices.extend(
            idx.numpy().tolist()
        )


        done = min(
            (
                batch_idx + 1
            )
            *
            BATCH_SIZE,
            len(dataset)
        )


        if (
            batch_idx == 0
            or
            done % 256 == 0
            or
            done == len(dataset)
        ):

            elapsed = (
                time.time()
                -
                start
            )


            print(
                f"{done}/"
                f"{len(dataset)}"
                f" | {elapsed:.1f}s"
            )


# =========================================================
# Stack
# =========================================================

X_low = np.concatenate(
    low_embeddings,
    axis=0
).astype(np.float32)


X_full = np.concatenate(
    full_embeddings,
    axis=0
).astype(np.float32)


indices = np.asarray(
    indices
)


# =========================================================
# Integrity
# =========================================================

print(
    "\nShapes:"
)

print(
    "Low:",
    X_low.shape
)

print(
    "Full:",
    X_full.shape
)


if (
    X_low.shape
    !=
    (
        len(manifest),
        EXPECTED_DIM
    )
):

    raise RuntimeError(
        f"Unexpected low shape: "
        f"{X_low.shape}"
    )


if (
    X_full.shape
    !=
    (
        len(manifest),
        EXPECTED_DIM
    )
):

    raise RuntimeError(
        f"Unexpected full shape: "
        f"{X_full.shape}"
    )


if not np.array_equal(
    indices,
    np.arange(
        len(manifest)
    )
):

    raise RuntimeError(
        "DataLoader ordering mismatch."
    )


# =========================================================
# Numerical QC
# =========================================================

low_nan = int(
    np.isnan(
        X_low
    ).sum()
)

full_nan = int(
    np.isnan(
        X_full
    ).sum()
)

low_inf = int(
    np.isinf(
        X_low
    ).sum()
)

full_inf = int(
    np.isinf(
        X_full
    ).sum()
)


low_norm = np.linalg.norm(
    X_low,
    axis=1
)

full_norm = np.linalg.norm(
    X_full,
    axis=1
)


paired_cosine = np.sum(
    X_low
    *
    X_full,
    axis=1
) / (
    low_norm
    *
    full_norm
    +
    1e-8
)


print(
    "\nNumerical QC:"
)

print(
    "Low NaN:",
    low_nan
)

print(
    "Full NaN:",
    full_nan
)

print(
    "Low Inf:",
    low_inf
)

print(
    "Full Inf:",
    full_inf
)

print(
    "Low mean L2:",
    float(
        low_norm.mean()
    )
)

print(
    "Full mean L2:",
    float(
        full_norm.mean()
    )
)

print(
    "Mean paired cosine:",
    float(
        paired_cosine.mean()
    )
)


if (
    low_nan
    or
    full_nan
    or
    low_inf
    or
    full_inf
):

    raise RuntimeError(
        "Invalid embedding values."
    )


# =========================================================
# Save
# =========================================================

np.save(
    LOW_OUT,
    X_low
)


np.save(
    FULL_OUT,
    X_full
)


embed_manifest = manifest[
    [
        "barcode",
        "fullres_x",
        "fullres_y",
        "hires_x",
        "hires_y",
        "lowres_patch_path",
        "fullres_patch_path",
    ]
].copy()


embed_manifest[
    "embedding_row"
] = np.arange(
    len(
        embed_manifest
    )
)


embed_manifest.to_csv(
    EMBED_MANIFEST_OUT,
    index=False
)


summary = pd.DataFrame(
    [
        {
            "scale":
                "lowres",

            "model":
                MODEL_NAME,

            "n_spots":
                X_low.shape[
                    0
                ],

            "embedding_dim":
                X_low.shape[
                    1
                ],

            "dtype":
                str(
                    X_low.dtype
                ),

            "nan_count":
                low_nan,

            "inf_count":
                low_inf,

            "feature_mean":
                float(
                    X_low.mean()
                ),

            "feature_std":
                float(
                    X_low.std()
                ),

            "mean_l2_norm":
                float(
                    low_norm.mean()
                ),

            "mean_paired_cosine":
                float(
                    paired_cosine.mean()
                ),
        },

        {
            "scale":
                "fullres",

            "model":
                MODEL_NAME,

            "n_spots":
                X_full.shape[
                    0
                ],

            "embedding_dim":
                X_full.shape[
                    1
                ],

            "dtype":
                str(
                    X_full.dtype
                ),

            "nan_count":
                full_nan,

            "inf_count":
                full_inf,

            "feature_mean":
                float(
                    X_full.mean()
                ),

            "feature_std":
                float(
                    X_full.std()
                ),

            "mean_l2_norm":
                float(
                    full_norm.mean()
                ),

            "mean_paired_cosine":
                float(
                    paired_cosine.mean()
                ),
        },
    ]
)


summary.to_csv(
    SUMMARY_OUT,
    index=False
)


elapsed = (
    time.time()
    -
    start
)


print(
    "\n======================================"
)

print(
    "D11 DINOV2 EXTRACTION COMPLETE"
)

print(
    "======================================"
)


print(
    "Low:",
    X_low.shape
)

print(
    "Full:",
    X_full.shape
)

print(
    "Elapsed:",
    f"{elapsed:.1f}s"
)


print(
    "\nSaved:"
)

print(
    LOW_OUT
)

print(
    FULL_OUT
)

print(
    EMBED_MANIFEST_OUT
)

print(
    SUMMARY_OUT
)
