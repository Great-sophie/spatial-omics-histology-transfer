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
    "D10_mouse_brain_external_validation"
)

OUTDIR = ROOT / "outputs"

MANIFEST_PATH = (
    OUTDIR
    / "external_multiscale_patch_manifest.csv"
)

LOW_OUT = (
    OUTDIR
    / "dinov2_vits14_external_lowres_embeddings.npy"
)

FULL_OUT = (
    OUTDIR
    / "dinov2_vits14_external_fullres_embeddings.npy"
)

EMBED_MANIFEST_OUT = (
    OUTDIR
    / "dinov2_vits14_external_embedding_manifest.csv"
)

SUMMARY_OUT = (
    OUTDIR
    / "dinov2_vits14_external_embedding_summary.csv"
)


MODEL_NAME = "dinov2_vits14"

# One item contains a low/full pair.
# BATCH_SIZE=16 means 32 images are forwarded together.
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
# Load manifest
# =========================================================

print("\nLoading patch manifest...")


manifest = pd.read_csv(
    MANIFEST_PATH
)


manifest["barcode"] = (
    manifest["barcode"]
    .astype(str)
)


required_columns = [
    "barcode",
    "lowres_patch_path",
    "fullres_patch_path",
]


for col in required_columns:

    if col not in manifest.columns:

        raise KeyError(
            f"Missing column: {col}"
        )


print(
    "Spots:",
    len(manifest)
)


# =========================================================
# File validation
# =========================================================

missing_low = []

missing_full = []


for _, row in manifest.iterrows():

    low_path = Path(
        row["lowres_patch_path"]
    )

    full_path = Path(
        row["fullres_patch_path"]
    )


    if not low_path.exists():

        missing_low.append(
            str(low_path)
        )


    if not full_path.exists():

        missing_full.append(
            str(full_path)
        )


print(
    "Missing low-res patches:",
    len(missing_low)
)

print(
    "Missing full-res patches:",
    len(missing_full)
)


if missing_low or missing_full:

    raise FileNotFoundError(
        "Patch files are incomplete."
    )


# =========================================================
# DINOv2 preprocessing
#
# Same ImageNet normalization used for D9.
# =========================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
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

class PairedPatchDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform,
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
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


dataset = PairedPatchDataset(
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
# Feature extraction helper
#
# DINOv2 forward_features returns:
# x_norm_clstoken
# =========================================================

def get_cls_embedding(
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
                "x_norm_clstoken "
                "not found in DINOv2 output."
            )


        emb = features[
            "x_norm_clstoken"
        ]


    else:

        emb = features


    return emb


# =========================================================
# Extract paired embeddings
# =========================================================

print(
    "\nExtracting paired DINOv2 embeddings..."
)


low_embeddings = []

full_embeddings = []

observed_indices = []


start_time = time.time()


with torch.inference_mode():

    for batch_idx, (
        low,
        full,
        idx
    ) in enumerate(loader):


        low = low.to(
            device,
            non_blocking=True,
        )


        full = full.to(
            device,
            non_blocking=True,
        )


        # Process both scales in one forward pass.
        images = torch.cat(
            [
                low,
                full,
            ],
            dim=0,
        )


        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                emb = get_cls_embedding(
                    images
                )

        else:

            emb = get_cls_embedding(
                images
            )


        emb = (
            emb
            .float()
            .cpu()
            .numpy()
        )


        n_batch = low.shape[
            0
        ]


        low_emb = emb[
            :n_batch
        ]


        full_emb = emb[
            n_batch:
        ]


        low_embeddings.append(
            low_emb
        )


        full_embeddings.append(
            full_emb
        )


        observed_indices.extend(
            idx.numpy().tolist()
        )


        done = min(
            (batch_idx + 1)
            *
            BATCH_SIZE,
            len(dataset),
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
                start_time
            )


            print(
                f"{done}/"
                f"{len(dataset)}"
                f" | elapsed "
                f"{elapsed:.1f}s"
            )


# =========================================================
# Concatenate
# =========================================================

X_low = np.concatenate(
    low_embeddings,
    axis=0,
).astype(np.float32)


X_full = np.concatenate(
    full_embeddings,
    axis=0,
).astype(np.float32)


observed_indices = np.asarray(
    observed_indices
)


# =========================================================
# Strict integrity checks
# =========================================================

print(
    "\nEmbedding shapes:"
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
    X_low.shape[0]
    !=
    len(manifest)
):

    raise RuntimeError(
        "Low embedding row count mismatch."
    )


if (
    X_full.shape[0]
    !=
    len(manifest)
):

    raise RuntimeError(
        "Full embedding row count mismatch."
    )


if (
    X_low.shape[1]
    !=
    EXPECTED_DIM
):

    raise RuntimeError(
        f"Unexpected low embedding dim: "
        f"{X_low.shape[1]}"
    )


if (
    X_full.shape[1]
    !=
    EXPECTED_DIM
):

    raise RuntimeError(
        f"Unexpected full embedding dim: "
        f"{X_full.shape[1]}"
    )


expected_indices = np.arange(
    len(manifest)
)


if not np.array_equal(
    observed_indices,
    expected_indices
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
    axis=1,
)


full_norm = np.linalg.norm(
    X_full,
    axis=1,
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
        "Invalid embedding values detected."
    )


# =========================================================
# Save embeddings
# =========================================================

np.save(
    LOW_OUT,
    X_low,
)


np.save(
    FULL_OUT,
    X_full,
)


# =========================================================
# Save aligned embedding manifest
# =========================================================

embedding_manifest = manifest[
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


embedding_manifest[
    "embedding_row"
] = np.arange(
    len(
        embedding_manifest
    )
)


embedding_manifest.to_csv(
    EMBED_MANIFEST_OUT,
    index=False,
)


# =========================================================
# Summary
# =========================================================

summary = pd.DataFrame(
    [
        {
            "scale":
                "lowres",

            "model":
                MODEL_NAME,

            "n_spots":
                X_low.shape[0],

            "embedding_dim":
                X_low.shape[1],

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

            "std_l2_norm":
                float(
                    low_norm.std()
                ),
        },

        {
            "scale":
                "fullres",

            "model":
                MODEL_NAME,

            "n_spots":
                X_full.shape[0],

            "embedding_dim":
                X_full.shape[1],

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

            "std_l2_norm":
                float(
                    full_norm.std()
                ),
        },
    ]
)


summary.to_csv(
    SUMMARY_OUT,
    index=False,
)


elapsed = (
    time.time()
    -
    start_time
)


# =========================================================
# Final report
# =========================================================

print(
    "\n========================================"
)

print(
    "D10 DINOV2 EXTRACTION COMPLETE"
)

print(
    "========================================"
)

print(
    "Low embedding:",
    X_low.shape
)

print(
    "Full embedding:",
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
