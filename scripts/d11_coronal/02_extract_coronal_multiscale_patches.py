import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps


# =========================================================
# Configuration
# =========================================================

ROOT = Path(
    "/media/sophie/Data2T/Biomedical AI  skill/"
    "D11_mouse_brain_coronal_validation"
)

DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

LOW_DIR = ROOT / "patches_lowres"
FULL_DIR = ROOT / "patches_fullres"

LOW_DIR.mkdir(exist_ok=True)
FULL_DIR.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


SAMPLE = "Visium_Fresh_Frozen_Adult_Mouse_Brain"

MANIFEST_PATH = (
    OUT
    / "coronal_spot_manifest.csv"
)

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

manifest = pd.read_csv(
    MANIFEST_PATH
)

manifest["barcode"] = (
    manifest["barcode"]
    .astype(str)
)


with open(
    SCALE_PATH
) as f:

    scales = json.load(f)


spot_diameter = float(
    scales[
        "spot_diameter_fullres"
    ]
)

hires_scale = float(
    scales[
        "tissue_hires_scalef"
    ]
)


# =========================================================
# Matched FOV
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


FULL_CROP = int(
    np.floor(
        full_fov_exact
        +
        0.5
    )
)

LOW_CROP = int(
    np.floor(
        hires_fov_exact
        +
        0.5
    )
)


print(
    "Spots:",
    len(manifest)
)

print(
    "spot_diameter_fullres:",
    spot_diameter
)

print(
    "tissue_hires_scalef:",
    hires_scale
)

print(
    "Full exact FOV:",
    full_fov_exact
)

print(
    "Full crop:",
    FULL_CROP
)

print(
    "Hires exact FOV:",
    hires_fov_exact
)

print(
    "Low crop:",
    LOW_CROP
)


# =========================================================
# Load images
# =========================================================

Image.MAX_IMAGE_PIXELS = None


hires = Image.open(
    HIRES_PATH
).convert("RGB")


full = Image.open(
    FULLRES_PATH
).convert("RGB")


print(
    "\nHires image:",
    hires.size
)

print(
    "Full image:",
    full.size
)


# =========================================================
# Safe centered crop
# =========================================================

def safe_crop(
    image,
    cx,
    cy,
    crop_size,
    output_size,
):

    cx = int(
        round(
            float(cx)
        )
    )

    cy = int(
        round(
            float(cy)
        )
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
    )


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


    patch = patch.convert(
        "RGB"
    )


    if (
        patch.size
        !=
        (
            output_size,
            output_size
        )
    ):

        patch = patch.resize(
            (
                output_size,
                output_size
            ),
            Image.Resampling.LANCZOS
        )


    return (
        patch,
        padded
    )


# =========================================================
# Coordinate QC
# =========================================================

x_full = (
    manifest[
        "fullres_x"
    ]
    .astype(float)
    .to_numpy()
)

y_full = (
    manifest[
        "fullres_y"
    ]
    .astype(float)
    .to_numpy()
)


outside_full = (
    (x_full < 0)
    |
    (x_full >= full.width)
    |
    (y_full < 0)
    |
    (y_full >= full.height)
)


print(
    "\nFull-res centers outside:",
    int(
        outside_full.sum()
    )
)


if outside_full.any():

    raise RuntimeError(
        "Full-resolution coordinates "
        "do not match TIFF."
    )


x_hires = (
    x_full
    *
    hires_scale
)

y_hires = (
    y_full
    *
    hires_scale
)


outside_hires = (
    (x_hires < 0)
    |
    (x_hires >= hires.width)
    |
    (y_hires < 0)
    |
    (y_hires >= hires.height)
)


print(
    "Hires centers outside:",
    int(
        outside_hires.sum()
    )
)


if outside_hires.any():

    raise RuntimeError(
        "Hires coordinates "
        "do not match image."
    )


# =========================================================
# Extract
# =========================================================

records = []

low_padded = 0
full_padded = 0


print(
    "\nExtracting patches..."
)


for i, row in manifest.iterrows():

    barcode = str(
        row[
            "barcode"
        ]
    )


    fx = float(
        row[
            "fullres_x"
        ]
    )

    fy = float(
        row[
            "fullres_y"
        ]
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


    # -----------------------------------------------------
    # Low resolution
    # -----------------------------------------------------

    low_patch, lp = safe_crop(
        image=hires,
        cx=hx,
        cy=hy,
        crop_size=LOW_CROP,
        output_size=OUTPUT_SIZE,
    )


    low_path = (
        LOW_DIR
        /
        f"{barcode}.png"
    )


    low_patch.save(
        low_path
    )


    # -----------------------------------------------------
    # Full resolution
    # -----------------------------------------------------

    full_patch, fp = safe_crop(
        image=full,
        cx=fx,
        cy=fy,
        crop_size=FULL_CROP,
        output_size=OUTPUT_SIZE,
    )


    full_path = (
        FULL_DIR
        /
        f"{barcode}.png"
    )


    full_patch.save(
        full_path
    )


    low_padded += int(
        lp
    )

    full_padded += int(
        fp
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
                LOW_CROP,

            "fullres_crop_size_px":
                FULL_CROP,

            "output_size_px":
                OUTPUT_SIZE,

            "lowres_patch_path":
                str(
                    low_path
                ),

            "fullres_patch_path":
                str(
                    full_path
                ),

            "lowres_padded":
                bool(
                    lp
                ),

            "fullres_padded":
                bool(
                    fp
                ),
        }
    )


    if (
        (i + 1) % 250 == 0
        or
        (i + 1)
        ==
        len(manifest)
    ):

        print(
            f"{i + 1}/"
            f"{len(manifest)}"
        )


# =========================================================
# Save manifest
# =========================================================

patch_manifest = pd.DataFrame(
    records
)


patch_manifest.to_csv(
    OUT
    / "coronal_multiscale_patch_manifest.csv",
    index=False
)


# =========================================================
# Summary
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
            len(
                patch_manifest
            ),

            spot_diameter,
            PATCH_FACTOR,
            full_fov_exact,
            FULL_CROP,
            hires_scale,
            hires_fov_exact,
            LOW_CROP,
            OUTPUT_SIZE,
            low_padded,
            full_padded,
            hires.width,
            hires.height,
            full.width,
            full.height,
        ],
    }
)


summary.to_csv(
    OUT
    / "coronal_patch_extraction_summary.csv",
    index=False
)


# =========================================================
# Contact sheet
# =========================================================

def make_contact_sheet(
    path_column,
    title,
    filename,
):

    rng = np.random.default_rng(
        42
    )


    n = min(
        36,
        len(
            patch_manifest
        )
    )


    chosen = rng.choice(
        len(
            patch_manifest
        ),
        size=n,
        replace=False
    )


    fig, axes = plt.subplots(
        6,
        6,
        figsize=(12, 12)
    )


    axes = axes.ravel()


    for ax, idx in zip(
        axes,
        chosen
    ):

        p = patch_manifest.iloc[
            idx
        ][
            path_column
        ]


        img = Image.open(
            p
        )


        ax.imshow(
            img
        )


        ax.axis(
            "off"
        )


    for ax in axes[
        n:
    ]:

        ax.axis(
            "off"
        )


    fig.suptitle(
        title
    )


    plt.tight_layout()


    plt.savefig(
        FIG
        /
        filename,
        dpi=250,
        bbox_inches="tight"
    )


    plt.close()


make_contact_sheet(
    "lowres_patch_path",
    (
        f"D11 Low-resolution patches "
        f"({LOW_CROP}px hires -> 224)"
    ),
    "03_coronal_lowres_patch_contact_sheet.png",
)


make_contact_sheet(
    "fullres_patch_path",
    (
        f"D11 Full-resolution patches "
        f"({FULL_CROP}px native -> 224)"
    ),
    "04_coronal_fullres_patch_contact_sheet.png",
)


# =========================================================
# Paired comparison
# =========================================================

rng = np.random.default_rng(
    123
)


n_show = min(
    12,
    len(
        patch_manifest
    )
)


chosen = rng.choice(
    len(
        patch_manifest
    ),
    size=n_show,
    replace=False
)


fig, axes = plt.subplots(
    n_show,
    2,
    figsize=(
        7,
        n_show * 3
    )
)


for r, idx in enumerate(
    chosen
):

    row = patch_manifest.iloc[
        idx
    ]


    low = Image.open(
        row[
            "lowres_patch_path"
        ]
    )


    high = Image.open(
        row[
            "fullres_patch_path"
        ]
    )


    axes[
        r,
        0
    ].imshow(
        low
    )


    axes[
        r,
        1
    ].imshow(
        high
    )


    axes[
        r,
        0
    ].set_title(
        "Low-res"
    )


    axes[
        r,
        1
    ].set_title(
        "Full-res"
    )


    axes[
        r,
        0
    ].axis(
        "off"
    )


    axes[
        r,
        1
    ].axis(
        "off"
    )


plt.tight_layout()


plt.savefig(
    FIG
    / "05_coronal_low_full_paired_patches.png",
    dpi=250,
    bbox_inches="tight"
)


plt.close()


# =========================================================
# Final size QC
# =========================================================

def inspect_folder(
    folder
):

    paths = list(
        folder.glob(
            "*.png"
        )
    )


    sizes = {}


    for p in paths:

        size = Image.open(
            p
        ).size


        sizes[
            size
        ] = (
            sizes.get(
                size,
                0
            )
            +
            1
        )


    return (
        len(paths),
        sizes
    )


n_low, low_sizes = inspect_folder(
    LOW_DIR
)


n_full, full_sizes = inspect_folder(
    FULL_DIR
)


print(
    "\n======================================"
)

print(
    "D11 MULTISCALE PATCHES COMPLETE"
)

print(
    "======================================"
)


print(
    "Spots:",
    len(
        patch_manifest
    )
)

print(
    "Low crop:",
    LOW_CROP,
    "hires px -> 224"
)

print(
    "Full crop:",
    FULL_CROP,
    "native px -> 224"
)

print(
    "Low padded:",
    low_padded
)

print(
    "Full padded:",
    full_padded
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
