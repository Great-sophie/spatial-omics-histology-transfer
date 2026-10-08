import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================

ROOT = Path(
    "/media/sophie/Data2T/Biomedical AI  skill/"
    "D10_mouse_brain_external_validation"
)

DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

LOW_DIR = ROOT / "patches_lowres"
FULL_DIR = ROOT / "patches_fullres"

OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
LOW_DIR.mkdir(exist_ok=True)
FULL_DIR.mkdir(exist_ok=True)


SAMPLE = "V1_Mouse_Brain_Sagittal_Anterior"

MANIFEST_PATH = OUT / "external_spot_manifest.csv"

HIRES_PATH = (
    DATA
    / "spatial"
    / "tissue_hires_image.png"
)

FULLRES_PATH = (
    DATA
    / f"{SAMPLE}_image.tif"
)

SCALE_PATH = (
    DATA
    / "spatial"
    / "scalefactors_json.json"
)


PATCH_FACTOR = 2.5
OUTPUT_SIZE = 224


# =========================================================
# Load metadata
# =========================================================

print("Loading manifest...")

manifest = pd.read_csv(
    MANIFEST_PATH
)

manifest["barcode"] = (
    manifest["barcode"]
    .astype(str)
)

print("Spots:", len(manifest))


with open(SCALE_PATH) as f:
    scale = json.load(f)


spot_diameter = float(
    scale["spot_diameter_fullres"]
)

hires_scale = float(
    scale["tissue_hires_scalef"]
)


# =========================================================
# Matched physical FOV
# =========================================================

full_fov_exact = (
    spot_diameter
    *
    PATCH_FACTOR
)

hires_fov_exact = (
    full_fov_exact
    *
    hires_scale
)


# round-half-up
full_crop = int(
    np.floor(
        full_fov_exact + 0.5
    )
)

low_crop = int(
    np.floor(
        hires_fov_exact + 0.5
    )
)


print("\nScale information")
print("-----------------")
print(
    "spot_diameter_fullres =",
    spot_diameter
)
print(
    "tissue_hires_scalef   =",
    hires_scale
)
print(
    "matched full FOV      =",
    full_fov_exact
)
print(
    "full-res crop         =",
    full_crop
)
print(
    "matched hires FOV     =",
    hires_fov_exact
)
print(
    "low-res crop          =",
    low_crop
)


# =========================================================
# Load images
# =========================================================

Image.MAX_IMAGE_PIXELS = None

print("\nLoading images...")

hires_img = Image.open(
    HIRES_PATH
).convert("RGB")

full_img = Image.open(
    FULLRES_PATH
)


print(
    "Hires:",
    hires_img.size
)

print(
    "Full:",
    full_img.size
)


# =========================================================
# Safe crop
# =========================================================

def crop_center(
    image,
    cx,
    cy,
    crop_size,
    resize_to=None,
):

    cx = int(
        round(float(cx))
    )

    cy = int(
        round(float(cy))
    )

    left = (
        cx
        -
        crop_size // 2
    )

    top = (
        cy
        -
        crop_size // 2
    )

    right = (
        left
        +
        crop_size
    )

    bottom = (
        top
        +
        crop_size
    )


    src_left = max(
        left,
        0
    )

    src_top = max(
        top,
        0
    )

    src_right = min(
        right,
        image.width
    )

    src_bottom = min(
        bottom,
        image.height
    )


    patch = image.crop(
        (
            src_left,
            src_top,
            src_right,
            src_bottom,
        )
    ).convert("RGB")


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
        right - image.width
    )

    pad_bottom = max(
        0,
        bottom - image.height
    )


    padded = any(
        [
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
        ]
    )


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
                255,
            ),
        )


    if (
        resize_to is not None
        and
        patch.size
        !=
        (
            resize_to,
            resize_to
        )
    ):

        patch = patch.resize(
            (
                resize_to,
                resize_to
            ),
            Image.Resampling.LANCZOS,
        )


    return patch, padded


# =========================================================
# Coordinate checks
# =========================================================

x_full = (
    manifest["fullres_x"]
    .astype(float)
    .to_numpy()
)

y_full = (
    manifest["fullres_y"]
    .astype(float)
    .to_numpy()
)


outside_full = (
    (x_full < 0)
    |
    (x_full >= full_img.width)
    |
    (y_full < 0)
    |
    (y_full >= full_img.height)
)


print(
    "\nFull-res centers outside image:",
    int(outside_full.sum())
)


if outside_full.any():

    raise RuntimeError(
        "Full-resolution coordinates "
        "do not align with TIFF."
    )


x_low = (
    x_full
    *
    hires_scale
)

y_low = (
    y_full
    *
    hires_scale
)


outside_low = (
    (x_low < 0)
    |
    (x_low >= hires_img.width)
    |
    (y_low < 0)
    |
    (y_low >= hires_img.height)
)


print(
    "Hires centers outside image:",
    int(outside_low.sum())
)


if outside_low.any():

    raise RuntimeError(
        "Hires coordinates "
        "do not align with hires image."
    )


# =========================================================
# Extract
# =========================================================

print(
    "\nExtracting multiscale patches..."
)

records = []

low_pad_count = 0
full_pad_count = 0


for i, row in manifest.iterrows():

    barcode = str(
        row["barcode"]
    )

    fx = float(
        row["fullres_x"]
    )

    fy = float(
        row["fullres_y"]
    )

    hx = (
        fx
        *
        hires_scale
    )

    hy = (
        fy
        *
        hires_scale
    )


    # ---------------------------------------------
    # Low-resolution branch
    # ---------------------------------------------

    low_patch, low_padded = crop_center(
        hires_img,
        hx,
        hy,
        low_crop,
        resize_to=OUTPUT_SIZE,
    )

    low_path = (
        LOW_DIR
        /
        f"{barcode}.png"
    )

    low_patch.save(
        low_path
    )


    # ---------------------------------------------
    # Full-resolution branch
    # ---------------------------------------------

    full_patch, full_padded = crop_center(
        full_img,
        fx,
        fy,
        full_crop,
        resize_to=OUTPUT_SIZE,
    )

    full_path = (
        FULL_DIR
        /
        f"{barcode}.png"
    )

    full_patch.save(
        full_path
    )


    low_pad_count += int(
        low_padded
    )

    full_pad_count += int(
        full_padded
    )


    records.append(
        {
            **row.to_dict(),

            "hires_x":
                hx,

            "hires_y":
                hy,

            "spot_diameter_fullres":
                spot_diameter,

            "patch_factor":
                PATCH_FACTOR,

            "lowres_crop_size_hires_px":
                low_crop,

            "fullres_crop_size_px":
                full_crop,

            "output_size_px":
                OUTPUT_SIZE,

            "lowres_patch_path":
                str(low_path),

            "fullres_patch_path":
                str(full_path),

            "lowres_padded":
                low_padded,

            "fullres_padded":
                full_padded,
        }
    )


    if (
        (i + 1) % 250 == 0
        or
        (i + 1) == len(manifest)
    ):

        print(
            f"{i+1}/{len(manifest)}"
        )


# =========================================================
# Save manifest
# =========================================================

patch_manifest = pd.DataFrame(
    records
)


patch_manifest.to_csv(
    OUT
    /
    "external_multiscale_patch_manifest.csv",
    index=False,
)


# =========================================================
# Save summary
# =========================================================

summary = pd.DataFrame(
    {
        "metric": [
            "n_spots",
            "spot_diameter_fullres",
            "patch_factor",
            "fullres_fov_exact",
            "fullres_crop_size_px",
            "tissue_hires_scalef",
            "hires_fov_exact",
            "lowres_crop_size_hires_px",
            "output_size_px",
            "lowres_padded_patches",
            "fullres_padded_patches",
            "hires_width",
            "hires_height",
            "fullres_width",
            "fullres_height",
        ],

        "value": [
            len(patch_manifest),
            spot_diameter,
            PATCH_FACTOR,
            full_fov_exact,
            full_crop,
            hires_scale,
            hires_fov_exact,
            low_crop,
            OUTPUT_SIZE,
            low_pad_count,
            full_pad_count,
            hires_img.width,
            hires_img.height,
            full_img.width,
            full_img.height,
        ],
    }
)


summary.to_csv(
    OUT
    /
    "external_patch_extraction_summary.csv",
    index=False,
)


# =========================================================
# Contact sheets
# =========================================================

def contact_sheet(
    paths,
    title,
    output_path,
):

    rng = np.random.default_rng(
        42
    )

    n = min(
        36,
        len(paths)
    )

    selected = rng.choice(
        len(paths),
        n,
        replace=False,
    )


    fig, axes = plt.subplots(
        6,
        6,
        figsize=(12, 12),
    )

    axes = axes.ravel()


    for ax, idx in zip(
        axes,
        selected
    ):

        img = Image.open(
            paths[idx]
        )

        ax.imshow(img)

        ax.axis("off")


    for ax in axes[n:]:

        ax.axis("off")


    fig.suptitle(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()


contact_sheet(
    patch_manifest[
        "lowres_patch_path"
    ].tolist(),

    (
        f"D10 Low-resolution patches "
        f"({low_crop}px -> 224px)"
    ),

    FIG
    /
    "03_external_lowres_patch_contact_sheet.png",
)


contact_sheet(
    patch_manifest[
        "fullres_patch_path"
    ].tolist(),

    (
        f"D10 Full-resolution patches "
        f"({full_crop}px)"
    ),

    FIG
    /
    "04_external_fullres_patch_contact_sheet.png",
)


# =========================================================
# Paired comparison
# =========================================================

rng = np.random.default_rng(
    123
)

n_show = min(
    12,
    len(patch_manifest)
)

selected = rng.choice(
    len(patch_manifest),
    n_show,
    replace=False,
)


fig, axes = plt.subplots(
    n_show,
    2,
    figsize=(7, n_show * 3),
)


for r, idx in enumerate(
    selected
):

    row = patch_manifest.iloc[
        idx
    ]

    low_img = Image.open(
        row["lowres_patch_path"]
    )

    full_img_patch = Image.open(
        row["fullres_patch_path"]
    )


    axes[r, 0].imshow(
        low_img
    )

    axes[r, 1].imshow(
        full_img_patch
    )


    axes[r, 0].set_title(
        "Low-res"
    )

    axes[r, 1].set_title(
        "Full-res"
    )


    axes[r, 0].axis(
        "off"
    )

    axes[r, 1].axis(
        "off"
    )


plt.tight_layout()

plt.savefig(
    FIG
    /
    "05_external_low_full_paired_patches.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# Verify output sizes
# =========================================================

def count_sizes(
    folder
):

    result = {}

    paths = list(
        folder.glob("*.png")
    )

    for p in paths:

        size = Image.open(p).size

        result[size] = (
            result.get(
                size,
                0
            )
            +
            1
        )

    return (
        len(paths),
        result,
    )


n_low, low_sizes = count_sizes(
    LOW_DIR
)

n_full, full_sizes = count_sizes(
    FULL_DIR
)


print(
    "\n========================================"
)

print(
    "D10 MULTISCALE PATCH EXTRACTION COMPLETE"
)

print(
    "========================================"
)

print(
    "Spots:",
    len(patch_manifest)
)

print(
    "Low crop:",
    low_crop,
    "hires px -> 224"
)

print(
    "Full crop:",
    full_crop,
    "true px -> 224"
)

print(
    "Low padded:",
    low_pad_count
)

print(
    "Full padded:",
    full_pad_count
)

print(
    "Low files:",
    n_low
)

print(
    "Full files:",
    n_full
)

print(
    "Low sizes:",
    low_sizes
)

print(
    "Full sizes:",
    full_sizes
)

print(
    "\nSaved:"
)

print(
    OUT
    /
    "external_multiscale_patch_manifest.csv"
)

print(
    OUT
    /
    "external_patch_extraction_summary.csv"
)

print(
    FIG
    /
    "03_external_lowres_patch_contact_sheet.png"
)

print(
    FIG
    /
    "04_external_fullres_patch_contact_sheet.png"
)

print(
    FIG
    /
    "05_external_low_full_paired_patches.png"
)
