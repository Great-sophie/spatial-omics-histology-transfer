import scanpy as sc
import pandas as pd
import numpy as np

from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# =========================================================
# Configuration
# =========================================================

INPUT = Path(
    "/media/sophie/Data2T/Biomedical AI  skill/"
    "D8 Squidpy/data/anndata/visium_hne_adata.h5ad"
)

PATCH_DIR = Path("patches")
OUTDIR = Path("outputs")
FIGDIR = Path("figures")

PATCH_DIR.mkdir(exist_ok=True)
OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

# How much tissue around each Visium spot to include.
# 1.0 ≈ one spot diameter.
# 2.5 gives morphology/context around the spot.
PATCH_FACTOR = 2.5

# Resize patch for future CNN / DINO / UNI input.
OUTPUT_SIZE = 224


# =========================================================
# 1. Load AnnData
# =========================================================

print("Loading Visium AnnData...")

adata = sc.read_h5ad(INPUT)

print(adata)

if "spatial" not in adata.obsm:
    raise KeyError("adata.obsm['spatial'] not found")

if "spatial" not in adata.uns:
    raise KeyError("adata.uns['spatial'] not found")


# =========================================================
# 2. Identify library
# =========================================================

library_ids = list(adata.uns["spatial"].keys())

print("\nLibrary IDs:")
print(library_ids)

if len(library_ids) != 1:
    print(
        "WARNING: multiple libraries found. "
        "Using the first library."
    )

library_id = library_ids[0]

spatial_meta = adata.uns["spatial"][library_id]

print("\nUsing library:")
print(library_id)

print("\nSpatial metadata keys:")
print(spatial_meta.keys())


# =========================================================
# 3. Inspect available H&E images
# =========================================================

images = spatial_meta["images"]

print("\nAvailable images:")
print(images.keys())

if "hires" in images:
    image_key = "hires"
elif "lowres" in images:
    image_key = "lowres"
else:
    raise KeyError(
        "Neither 'hires' nor 'lowres' image found."
    )

img_array = images[image_key]

print("\nImage key:")
print(image_key)

print("Image shape:")
print(img_array.shape)

print("Image dtype:")
print(img_array.dtype)


# =========================================================
# 4. Convert image to uint8 PIL
# =========================================================

if img_array.dtype != np.uint8:

    if img_array.max() <= 1.0:
        img_uint8 = (
            np.clip(img_array, 0, 1) * 255
        ).astype(np.uint8)

    else:
        img_uint8 = np.clip(
            img_array,
            0,
            255,
        ).astype(np.uint8)

else:
    img_uint8 = img_array


image = Image.fromarray(img_uint8)

print("\nPIL image size:")
print(image.size)


# =========================================================
# 5. Scaling information
# =========================================================

scalefactors = spatial_meta["scalefactors"]

print("\nScale factors:")
print(scalefactors)

if image_key == "hires":
    scale = scalefactors["tissue_hires_scalef"]
else:
    scale = scalefactors["tissue_lowres_scalef"]


spot_diameter_fullres = scalefactors[
    "spot_diameter_fullres"
]

spot_diameter_image = (
    spot_diameter_fullres * scale
)

patch_size = int(
    round(
        spot_diameter_image
        * PATCH_FACTOR
    )
)

# force even number for symmetric crop
if patch_size % 2 != 0:
    patch_size += 1


print("\nCoordinate scale:")
print(scale)

print("Spot diameter full resolution:")
print(spot_diameter_fullres)

print("Spot diameter in selected image:")
print(spot_diameter_image)

print("Crop size:")
print(patch_size)

print("Model output size:")
print(OUTPUT_SIZE)


# =========================================================
# 6. Full-resolution coordinates → image coordinates
# =========================================================

coords_fullres = adata.obsm["spatial"].astype(float)

coords_image = coords_fullres * scale

print("\nCoordinates:")
print("full resolution:", coords_fullres.shape)
print("image resolution:", coords_image.shape)

print("\nFirst 5 transformed coordinates:")
print(coords_image[:5])


# =========================================================
# 7. Extract patches
# =========================================================

half = patch_size // 2

width, height = image.size

rows = []

print("\nExtracting patches...")

for i, barcode in enumerate(adata.obs_names):

    x = float(coords_image[i, 0])
    y = float(coords_image[i, 1])

    cx = int(round(x))
    cy = int(round(y))

    left = cx - half
    upper = cy - half
    right = cx + half
    lower = cy + half

    # Create white canvas in case patch touches image edge
    patch = Image.new(
        "RGB",
        (patch_size, patch_size),
        color=(255, 255, 255),
    )

    crop_left = max(0, left)
    crop_upper = max(0, upper)
    crop_right = min(width, right)
    crop_lower = min(height, lower)

    crop = image.crop(
        (
            crop_left,
            crop_upper,
            crop_right,
            crop_lower,
        )
    ).convert("RGB")

    paste_x = crop_left - left
    paste_y = crop_upper - upper

    patch.paste(
        crop,
        (
            paste_x,
            paste_y,
        )
    )

    # Resize to standard image-model input
    patch_224 = patch.resize(
        (OUTPUT_SIZE, OUTPUT_SIZE),
        Image.Resampling.LANCZOS,
    )

    filename = f"{barcode}.png"

    patch_path = PATCH_DIR / filename

    patch_224.save(patch_path)

    row = {
        "barcode": barcode,

        "array_row": (
            adata.obs.iloc[i]["array_row"]
            if "array_row" in adata.obs.columns
            else np.nan
        ),

        "array_col": (
            adata.obs.iloc[i]["array_col"]
            if "array_col" in adata.obs.columns
            else np.nan
        ),

        "fullres_x": coords_fullres[i, 0],
        "fullres_y": coords_fullres[i, 1],

        "image_x": x,
        "image_y": y,

        "patch_size_image_px": patch_size,
        "output_size_px": OUTPUT_SIZE,

        "patch_path": str(patch_path),
    }

    if "cluster" in adata.obs.columns:
        row["cluster"] = str(
            adata.obs.iloc[i]["cluster"]
        )

    if "leiden" in adata.obs.columns:
        row["leiden"] = str(
            adata.obs.iloc[i]["leiden"]
        )

    rows.append(row)


manifest = pd.DataFrame(rows)

manifest.to_csv(
    OUTDIR / "spot_patch_manifest.csv",
    index=False,
)


print("\nPatches extracted:")
print(len(manifest))

print("\nManifest:")
print(
    OUTDIR / "spot_patch_manifest.csv"
)


# =========================================================
# 8. QC overlay: draw spot positions on H&E
# =========================================================

overlay = image.convert("RGB").copy()

draw = ImageDraw.Draw(overlay)

radius = max(
    2,
    int(round(
        spot_diameter_image * 0.18
    ))
)

for x, y in coords_image:

    x = int(round(x))
    y = int(round(y))

    draw.ellipse(
        (
            x - radius,
            y - radius,
            x + radius,
            y + radius,
        ),
        outline=(255, 0, 0),
        width=1,
    )

overlay.save(
    FIGDIR / "01_spot_overlay_on_hne.png"
)


# =========================================================
# 9. Contact sheet of sample patches
# =========================================================

n_show = min(36, len(manifest))

indices = np.linspace(
    0,
    len(manifest) - 1,
    n_show,
    dtype=int,
)

thumb_size = 128

cols = 6
rows_n = int(
    np.ceil(n_show / cols)
)

sheet = Image.new(
    "RGB",
    (
        cols * thumb_size,
        rows_n * thumb_size,
    ),
    color=(255, 255, 255),
)

for j, idx in enumerate(indices):

    p = Image.open(
        manifest.iloc[idx]["patch_path"]
    ).convert("RGB")

    p = p.resize(
        (thumb_size, thumb_size)
    )

    col = j % cols
    row = j // cols

    sheet.paste(
        p,
        (
            col * thumb_size,
            row * thumb_size,
        )
    )


sheet.save(
    FIGDIR / "02_patch_contact_sheet.png"
)


# =========================================================
# 10. Summary
# =========================================================

summary = pd.DataFrame(
    {
        "metric": [
            "n_spots",
            "image_key",
            "image_width",
            "image_height",
            "coordinate_scale",
            "spot_diameter_fullres",
            "spot_diameter_image",
            "patch_factor",
            "crop_size_px",
            "model_input_size_px",
        ],

        "value": [
            adata.n_obs,
            image_key,
            width,
            height,
            scale,
            spot_diameter_fullres,
            spot_diameter_image,
            PATCH_FACTOR,
            patch_size,
            OUTPUT_SIZE,
        ],
    }
)

summary.to_csv(
    OUTDIR / "patch_extraction_summary.csv",
    index=False,
)


print("\nSaved:")
print("patches/*.png")
print("outputs/spot_patch_manifest.csv")
print("outputs/patch_extraction_summary.csv")
print("figures/01_spot_overlay_on_hne.png")
print("figures/02_patch_contact_sheet.png")

print("\nD9 patch extraction complete.")
