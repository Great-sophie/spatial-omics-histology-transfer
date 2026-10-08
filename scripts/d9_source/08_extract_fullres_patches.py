import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image, ImageOps


# =========================================================
# Configuration
# =========================================================

FULLRES_IMAGE = Path(
    "/media/sophie/Data2T/Biomedical AI  skill/"
    "D8 Squidpy/data/anndata/V1_Adult_Mouse_Brain_image.tif"
)

OLD_MANIFEST = Path(
    "outputs/spot_image_embedding_manifest.csv"
)

PATCH_DIR = Path(
    "patches_fullres"
)

OUTDIR = Path(
    "outputs"
)

FIGDIR = Path(
    "figures"
)

PATCH_DIR.mkdir(exist_ok=True)
OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)


# Same physical FOV as previous experiment:
#
# spot_diameter_fullres ≈ 89.44 px
# PATCH_FACTOR = 2.5
#
# 89.44 × 2.5 ≈ 223.6 px
#
# therefore:
PATCH_SIZE = 224

HALF_SIZE = PATCH_SIZE // 2


# =========================================================
# 1. Load manifest
# =========================================================

print("Loading manifest...")

manifest = pd.read_csv(
    OLD_MANIFEST
)

print(
    "Manifest shape:",
    manifest.shape
)


required_columns = [
    "barcode",
    "fullres_x",
    "fullres_y",
]

for col in required_columns:

    if col not in manifest.columns:

        raise KeyError(
            f"Required column missing: {col}"
        )


# =========================================================
# 2. Load full-resolution TIFF
# =========================================================

print(
    "\nOpening full-resolution H&E..."
)

Image.MAX_IMAGE_PIXELS = None

image = Image.open(
    FULLRES_IMAGE
).convert("RGB")


width, height = image.size


print(
    "Image size:",
    image.size
)

print(
    "Mode:",
    image.mode
)

print(
    "Patch size:",
    PATCH_SIZE,
    "×",
    PATCH_SIZE
)


# =========================================================
# 3. Coordinate QC
# =========================================================

x_coords = (
    manifest[
        "fullres_x"
    ]
    .astype(float)
    .values
)

y_coords = (
    manifest[
        "fullres_y"
    ]
    .astype(float)
    .values
)


print(
    "\nCoordinate range:"
)

print(
    "x:",
    float(x_coords.min()),
    "→",
    float(x_coords.max())
)

print(
    "y:",
    float(y_coords.min()),
    "→",
    float(y_coords.max())
)


outside = (
    (x_coords < 0)
    |
    (x_coords >= width)
    |
    (y_coords < 0)
    |
    (y_coords >= height)
)


print(
    "Spot centers outside image:",
    int(
        outside.sum()
    )
)


if outside.any():

    raise ValueError(
        "Some spot centers fall outside "
        "the full-resolution image."
    )


# =========================================================
# 4. Helper: safe crop with white padding
# =========================================================

def extract_patch(
    image,
    center_x,
    center_y,
    patch_size,
):

    half = (
        patch_size
        // 2
    )


    cx = int(
        round(
            float(center_x)
        )
    )

    cy = int(
        round(
            float(center_y)
        )
    )


    left = (
        cx
        - half
    )

    top = (
        cy
        - half
    )

    right = (
        left
        + patch_size
    )

    bottom = (
        top
        + patch_size
    )


    # Valid image region
    crop_left = max(
        0,
        left
    )

    crop_top = max(
        0,
        top
    )

    crop_right = min(
        image.width,
        right
    )

    crop_bottom = min(
        image.height,
        bottom
    )


    patch = image.crop(
        (
            crop_left,
            crop_top,
            crop_right,
            crop_bottom,
        )
    )


    # Padding required?
    pad_left = max(
        0,
        -left
    )

    pad_top = max(
        0,
        -top
    )

    pad_right = max(
        0,
        right
        - image.width
    )

    pad_bottom = max(
        0,
        bottom
        - image.height
    )


    padded = (
        pad_left
        + pad_top
        + pad_right
        + pad_bottom
    ) > 0


    if padded:

        patch = ImageOps.expand(
            patch,
            border=(
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ),
            fill=(
                255,
                255,
                255
            ),
        )


    # Defensive check
    if (
        patch.width
        != patch_size
        or
        patch.height
        != patch_size
    ):

        patch = patch.resize(
            (
                patch_size,
                patch_size
            ),
            Image.Resampling.LANCZOS
        )


    return (
        patch,
        padded
    )


# =========================================================
# 5. Extract patches
# =========================================================

print(
    "\nExtracting full-resolution patches..."
)


records = []

padded_count = 0


for i, row in manifest.iterrows():

    barcode = str(
        row[
            "barcode"
        ]
    )


    patch, padded = extract_patch(
        image=image,
        center_x=row[
            "fullres_x"
        ],
        center_y=row[
            "fullres_y"
        ],
        patch_size=PATCH_SIZE,
    )


    filename = (
        f"{barcode}.png"
    )


    output_path = (
        PATCH_DIR
        / filename
    )


    patch.save(
        output_path,
        format="PNG"
    )


    if padded:

        padded_count += 1


    record = row.to_dict()


    record[
        "fullres_patch_path"
    ] = str(
        output_path
    )


    record[
        "fullres_patch_size_px"
    ] = PATCH_SIZE


    record[
        "fullres_patch_padded"
    ] = bool(
        padded
    )


    records.append(
        record
    )


    if (
        i == 0
        or
        (i + 1) % 250 == 0
        or
        (i + 1) == len(manifest)
    ):

        print(
            f"{i + 1}/"
            f"{len(manifest)}"
        )


# =========================================================
# 6. Save new manifest
# =========================================================

fullres_manifest = pd.DataFrame(
    records
)


fullres_manifest.to_csv(
    OUTDIR
    / "fullres_spot_patch_manifest.csv",
    index=False,
)


print(
    "\nPadded patches:",
    padded_count
)


# =========================================================
# 7. Full-resolution coordinate overlay QC
#
# Generate thumbnail only for visualization.
# Original image remains untouched.
# =========================================================

print(
    "\nGenerating coordinate overlay..."
)


thumb_width = 2000

thumb_scale = (
    thumb_width
    / width
)

thumb_height = int(
    round(
        height
        * thumb_scale
    )
)


thumbnail = image.resize(
    (
        thumb_width,
        thumb_height
    ),
    Image.Resampling.LANCZOS
)


x_thumb = (
    x_coords
    * thumb_scale
)

y_thumb = (
    y_coords
    * thumb_scale
)


fig, ax = plt.subplots(
    figsize=(10, 10)
)


ax.imshow(
    thumbnail
)


ax.scatter(
    x_thumb,
    y_thumb,
    s=4,
    facecolors="none",
    edgecolors="red",
    linewidths=0.4,
)


ax.set_title(
    "Visium spots on full-resolution H&E"
)


ax.axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "32_fullres_spot_overlay.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 8. Contact sheet QC
# =========================================================

print(
    "Generating full-resolution patch contact sheet..."
)


rng = np.random.default_rng(
    42
)


n_show = min(
    36,
    len(
        fullres_manifest
    )
)


sample_indices = rng.choice(
    len(
        fullres_manifest
    ),
    size=n_show,
    replace=False,
)


fig, axes = plt.subplots(
    6,
    6,
    figsize=(
        12,
        12
    )
)


axes = axes.ravel()


for ax, idx in zip(
    axes,
    sample_indices
):

    patch_path = (
        fullres_manifest.iloc[
            idx
        ][
            "fullres_patch_path"
        ]
    )


    patch = Image.open(
        patch_path
    )


    ax.imshow(
        patch
    )


    barcode = (
        fullres_manifest.iloc[
            idx
        ][
            "barcode"
        ]
    )


    cluster = (
        fullres_manifest.iloc[
            idx
        ].get(
            "cluster",
            ""
        )
    )


    ax.set_title(
        f"{barcode[:8]}\n{cluster}",
        fontsize=6,
    )


    ax.axis(
        "off"
    )


for ax in axes[
    n_show:
]:

    ax.axis(
        "off"
    )


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "33_fullres_patch_contact_sheet.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 9. Summary
# =========================================================

summary = pd.DataFrame(
    {
        "metric": [
            "n_spots",
            "image_width",
            "image_height",
            "patch_size_px",
            "patch_area_px",
            "n_padded_patches",
            "coordinate_system",
            "source_image",
        ],

        "value": [
            len(
                fullres_manifest
            ),
            width,
            height,
            PATCH_SIZE,
            PATCH_SIZE
            * PATCH_SIZE,
            padded_count,
            "full_resolution",
            str(
                FULLRES_IMAGE
            ),
        ],
    }
)


summary.to_csv(
    OUTDIR
    / "fullres_patch_extraction_summary.csv",
    index=False,
)


# =========================================================
# 10. Final report
# =========================================================

print(
    "\nSaved:"
)

print(
    "patches_fullres/*.png"
)

print(
    "outputs/fullres_spot_patch_manifest.csv"
)

print(
    "outputs/fullres_patch_extraction_summary.csv"
)

print(
    "figures/32_fullres_spot_overlay.png"
)

print(
    "figures/33_fullres_patch_contact_sheet.png"
)


print(
    "\nFull-resolution patch extraction complete."
)
