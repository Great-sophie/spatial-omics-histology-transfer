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
    "outputs/fullres_spot_patch_manifest.csv"
)

OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

OUTPUT_EMBEDDING = (
    OUTDIR
    / "dinov2_vits14_fullres_image_embeddings.npy"
)

OUTPUT_MANIFEST = (
    OUTDIR
    / "dinov2_vits14_fullres_embedding_manifest.csv"
)

OUTPUT_SUMMARY = (
    OUTDIR
    / "dinov2_vits14_fullres_embedding_summary.csv"
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


print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================================================
# 1. Load manifest
# =========================================================

print(
    "\nLoading full-resolution patch manifest..."
)

manifest = pd.read_csv(
    MANIFEST_PATH
)

print(
    "Manifest:",
    manifest.shape
)


if "fullres_patch_path" not in manifest.columns:

    raise KeyError(
        "fullres_patch_path column not found."
    )


# =========================================================
# 2. Check patch files
# =========================================================

patch_paths = [
    Path(p)
    for p in manifest[
        "fullres_patch_path"
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
        f"{len(missing)} patches missing."
    )


print(
    "Number of patches:",
    len(patch_paths)
)

print(
    "All full-resolution patches found."
)


# =========================================================
# 3. DINOv2 preprocessing
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

        return len(self.paths)


    def __getitem__(
        self,
        idx,
    ):

        path = self.paths[idx]

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return image, idx


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
# Should use cached code/weights from script 06.
# =========================================================

print(
    "\nLoading DINOv2..."
)

print(
    "Model:",
    MODEL_NAME
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
# 6. Detect embedding dimension
# =========================================================

print(
    "\nTesting forward pass..."
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

            features = model.forward_features(
                test_images
            )

    else:

        features = model.forward_features(
            test_images
        )


    embeddings_test = (
        features[
            "x_norm_clstoken"
        ]
        .float()
        .cpu()
        .numpy()
    )


EMBEDDING_DIM = (
    embeddings_test.shape[1]
)


print(
    "Test embedding shape:",
    embeddings_test.shape
)

print(
    "Embedding dimension:",
    EMBEDDING_DIM
)


del test_images
del features
del embeddings_test


if torch.cuda.is_available():
    torch.cuda.empty_cache()


# =========================================================
# 7. Extract embeddings
# =========================================================

print(
    "\nExtracting full-resolution DINOv2 embeddings..."
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

                features = model.forward_features(
                    images
                )

                embeddings = features[
                    "x_norm_clstoken"
                ]

        else:

            features = model.forward_features(
                images
            )

            embeddings = features[
                "x_norm_clstoken"
            ]


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


embedding_norms = np.linalg.norm(
    all_embeddings,
    axis=1,
)


print(
    "\n=== Full-resolution embedding QC ==="
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
    "Mean L2 norm:",
    float(
        embedding_norms.mean()
    )
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
        "NaNs found in embeddings."
    )


if inf_count > 0:

    raise ValueError(
        "Infinite values found in embeddings."
    )


# =========================================================
# 9. Save embeddings
# =========================================================

np.save(
    OUTPUT_EMBEDDING,
    all_embeddings,
)


# =========================================================
# 10. Save manifest
# =========================================================

embedding_manifest = (
    manifest.copy()
)


embedding_manifest[
    "embedding_model"
] = MODEL_NAME


embedding_manifest[
    "embedding_resolution"
] = "fullres"


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
# 11. Summary
# =========================================================

summary = pd.DataFrame(
    {
        "metric": [
            "n_patches",
            "model",
            "resolution",
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
            "full_resolution",
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
# 12. Final report
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
    "\nFull-resolution DINOv2 embedding extraction complete."
)

print(
    "\nNo PCA was fitted."
)

print(
    "Final comparison must use train-region-only PCA."
)
