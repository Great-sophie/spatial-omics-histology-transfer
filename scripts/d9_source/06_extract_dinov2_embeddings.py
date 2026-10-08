import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =========================================================
# Configuration
# =========================================================

MANIFEST_PATH = Path(
    "outputs/spot_image_embedding_manifest.csv"
)

OUTDIR = Path("outputs")

OUTDIR.mkdir(exist_ok=True)


OUTPUT_EMBEDDING = (
    OUTDIR
    / "dinov2_vits14_image_embeddings.npy"
)

OUTPUT_MANIFEST = (
    OUTDIR
    / "dinov2_vits14_embedding_manifest.csv"
)

OUTPUT_SUMMARY = (
    OUTDIR
    / "dinov2_vits14_embedding_summary.csv"
)


MODEL_NAME = "dinov2_vits14"

BATCH_SIZE = 32

NUM_WORKERS = 4

IMAGE_SIZE = 224


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================================================
# 1. Load manifest
# =========================================================

print(
    "\nLoading patch manifest..."
)


manifest = pd.read_csv(
    MANIFEST_PATH
)


print(
    "Manifest:",
    manifest.shape
)


if "patch_path" not in manifest.columns:

    raise KeyError(
        "patch_path column not found "
        "in manifest."
    )


# =========================================================
# 2. Check patch files
# =========================================================

patch_paths = [
    Path(p)
    for p in manifest[
        "patch_path"
    ].astype(str)
]


missing = [
    str(p)
    for p in patch_paths
    if not p.exists()
]


if missing:

    print(
        "\nExample missing paths:"
    )

    for p in missing[:10]:

        print(p)

    raise FileNotFoundError(
        f"{len(missing)} patch files "
        "are missing."
    )


print(
    "Number of patches:",
    len(patch_paths)
)

print(
    "All patch files found."
)


# =========================================================
# 3. Image preprocessing
#
# DINOv2 commonly uses ImageNet normalization.
#
# Patches are already 224 × 224, but Resize is kept
# here for robustness.
# =========================================================

transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            antialias=True,
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
# 4. Dataset
# =========================================================

class PatchDataset(Dataset):

    def __init__(
        self,
        paths,
        transform,
    ):

        self.paths = paths

        self.transform = transform


    def __len__(self):

        return len(
            self.paths
        )


    def __getitem__(
        self,
        idx,
    ):

        path = self.paths[
            idx
        ]


        image = Image.open(
            path
        ).convert(
            "RGB"
        )


        image = self.transform(
            image
        )


        return (
            image,
            idx
        )


dataset = PatchDataset(
    patch_paths,
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
# 5. Load DINOv2
#
# First run may download:
# - facebookresearch/dinov2 repository
# - pretrained model weights
# =========================================================

print(
    "\nLoading DINOv2..."
)

print(
    "Model:",
    MODEL_NAME
)

print(
    "First run may download "
    "the pretrained weights."
)


model = torch.hub.load(
    "facebookresearch/dinov2",
    MODEL_NAME,
)


model = model.to(
    device
)


model.eval()


# =========================================================
# 6. Test one batch and detect embedding dimension
# =========================================================

print(
    "\nTesting DINOv2 forward pass..."
)


test_images, _ = next(
    iter(loader)
)


test_images = test_images.to(
    device,
    non_blocking=True,
)


with torch.inference_mode():

    if device.type == "cuda":

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            features = (
                model.forward_features(
                    test_images
                )
            )

    else:

        features = (
            model.forward_features(
                test_images
            )
        )


    if (
        "x_norm_clstoken"
        not in features
    ):

        raise KeyError(
            "DINOv2 forward_features() "
            "did not return "
            "'x_norm_clstoken'."
        )


    test_embedding = (
        features[
            "x_norm_clstoken"
        ]
        .float()
        .cpu()
        .numpy()
    )


EMBEDDING_DIM = (
    test_embedding.shape[1]
)


print(
    "Test embedding shape:",
    test_embedding.shape
)

print(
    "Embedding dimension:",
    EMBEDDING_DIM
)


del test_images
del features
del test_embedding

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# =========================================================
# 7. Extract embeddings
# =========================================================

print(
    "\nExtracting DINOv2 embeddings..."
)


all_embeddings = np.zeros(
    (
        len(dataset),
        EMBEDDING_DIM
    ),
    dtype=np.float32,
)


start_time = time.time()

processed = 0


with torch.inference_mode():

    for batch_index, (
        images,
        indices
    ) in enumerate(loader):


        images = images.to(
            device,
            non_blocking=True,
        )


        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                features = (
                    model.forward_features(
                        images
                    )
                )


                embeddings = (
                    features[
                        "x_norm_clstoken"
                    ]
                )

        else:

            features = (
                model.forward_features(
                    images
                )
            )


            embeddings = (
                features[
                    "x_norm_clstoken"
                ]
            )


        embeddings = (
            embeddings
            .float()
            .cpu()
            .numpy()
        )


        indices_np = (
            indices
            .cpu()
            .numpy()
        )


        all_embeddings[
            indices_np
        ] = embeddings


        processed += len(
            indices_np
        )


        if (
            batch_index == 0
            or
            processed % 320 < BATCH_SIZE
            or
            processed == len(dataset)
        ):

            elapsed = (
                time.time()
                -
                start_time
            )


            rate = (
                processed
                /
                max(
                    elapsed,
                    1e-6
                )
            )


            print(
                f"{processed}/"
                f"{len(dataset)}"
                f" | "
                f"{rate:.1f} patches/s"
            )


# =========================================================
# 8. QC
# =========================================================

elapsed = (
    time.time()
    -
    start_time
)


nan_count = int(
    np.isnan(
        all_embeddings
    ).sum()
)


inf_count = int(
    np.isinf(
        all_embeddings
    ).sum()
)


print(
    "\n=== Embedding QC ==="
)

print(
    "Shape:",
    all_embeddings.shape
)

print(
    "dtype:",
    all_embeddings.dtype
)

print(
    "Mean:",
    float(
        all_embeddings.mean()
    )
)

print(
    "Std:",
    float(
        all_embeddings.std()
    )
)

print(
    "NaN count:",
    nan_count
)

print(
    "Inf count:",
    inf_count
)

print(
    "Elapsed seconds:",
    round(
        elapsed,
        2
    )
)


if nan_count > 0:

    raise ValueError(
        "NaNs detected in embeddings."
    )


if inf_count > 0:

    raise ValueError(
        "Infinite values detected "
        "in embeddings."
    )


# =========================================================
# 9. Embedding norm QC
# =========================================================

embedding_norms = np.linalg.norm(
    all_embeddings,
    axis=1,
)


print(
    "\nEmbedding norm:"
)

print(
    "mean:",
    float(
        embedding_norms.mean()
    )
)

print(
    "std:",
    float(
        embedding_norms.std()
    )
)

print(
    "min:",
    float(
        embedding_norms.min()
    )
)

print(
    "max:",
    float(
        embedding_norms.max()
    )
)


# =========================================================
# 10. Save raw embeddings
#
# IMPORTANT:
# No global PCA is performed here.
#
# PCA for benchmarking must later be fitted
# using training regions only.
# =========================================================

np.save(
    OUTPUT_EMBEDDING,
    all_embeddings,
)


# =========================================================
# 11. Save aligned manifest
# =========================================================

embedding_manifest = (
    manifest.copy()
)


embedding_manifest[
    "embedding_model"
] = MODEL_NAME


embedding_manifest[
    "embedding_dimension"
] = EMBEDDING_DIM


embedding_manifest[
    "embedding_row"
] = np.arange(
    len(
        embedding_manifest
    )
)


embedding_manifest.to_csv(
    OUTPUT_MANIFEST,
    index=False,
)


# =========================================================
# 12. Save summary
# =========================================================

summary = pd.DataFrame(
    {
        "metric": [
            "n_patches",
            "model",
            "input_size",
            "embedding_dimension",
            "batch_size",
            "device",
            "embedding_dtype",
            "embedding_mean",
            "embedding_std",
            "embedding_norm_mean",
            "embedding_norm_std",
            "nan_count",
            "inf_count",
            "elapsed_seconds",
        ],

        "value": [
            len(dataset),
            MODEL_NAME,
            IMAGE_SIZE,
            EMBEDDING_DIM,
            BATCH_SIZE,
            str(device),
            str(
                all_embeddings.dtype
            ),
            float(
                all_embeddings.mean()
            ),
            float(
                all_embeddings.std()
            ),
            float(
                embedding_norms.mean()
            ),
            float(
                embedding_norms.std()
            ),
            nan_count,
            inf_count,
            elapsed,
        ],
    }
)


summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# =========================================================
# 13. Final report
# =========================================================

print(
    "\nSaved:"
)

print(
    OUTPUT_EMBEDDING
)

print(
    OUTPUT_MANIFEST
)

print(
    OUTPUT_SUMMARY
)


print(
    "\nDINOv2 embedding extraction complete."
)

print(
    "\nIMPORTANT:"
)

print(
    "Do NOT fit PCA on the full dataset "
    "for the final benchmark."
)

print(
    "The next strict comparison will fit "
    "PCA on training regions only."
)
