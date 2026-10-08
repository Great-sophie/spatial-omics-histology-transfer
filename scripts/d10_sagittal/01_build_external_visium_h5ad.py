import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# Configuration
# =========================================================

ROOT = Path(
    "/media/sophie/Data2T/Biomedical AI  skill/"
    "D10_mouse_brain_external_validation"
)

DATA_DIR = ROOT / "data"
OUTDIR = ROOT / "outputs"
FIGDIR = ROOT / "figures"

OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)


SAMPLE = "V1_Mouse_Brain_Sagittal_Anterior"


MATRIX_PATH = (
    DATA_DIR
    / f"{SAMPLE}_filtered_feature_bc_matrix.h5"
)

FULLRES_IMAGE_PATH = (
    DATA_DIR
    / f"{SAMPLE}_image.tif"
)

SPATIAL_DIR = (
    DATA_DIR
    / "spatial"
)

SCALEFACTOR_PATH = (
    SPATIAL_DIR
    / "scalefactors_json.json"
)

HIRES_IMAGE_PATH = (
    SPATIAL_DIR
    / "tissue_hires_image.png"
)

LOWRES_IMAGE_PATH = (
    SPATIAL_DIR
    / "tissue_lowres_image.png"
)


TARGET_GENES = [
    "Camk2n1",
    "Slc17a7",
    "Plp1",
    "Cldn11",
]


LIBRARY_ID = SAMPLE


# =========================================================
# 1. Check input files
# =========================================================

print("Checking input files...")


required_paths = [
    MATRIX_PATH,
    SCALEFACTOR_PATH,
    HIRES_IMAGE_PATH,
    LOWRES_IMAGE_PATH,
]


for p in required_paths:

    print(
        p,
        "FOUND" if p.exists() else "MISSING"
    )

    if not p.exists():

        raise FileNotFoundError(
            str(p)
        )


if FULLRES_IMAGE_PATH.exists():

    print(
        FULLRES_IMAGE_PATH,
        "FOUND"
    )

else:

    print(
        FULLRES_IMAGE_PATH,
        "NOT FOUND"
    )

    print(
        "AnnData can still be built, "
        "but full-resolution patch extraction "
        "will require this TIFF."
    )


# =========================================================
# 2. Find tissue positions file
#
# New Space Ranger:
# tissue_positions.csv
#
# Older Space Ranger:
# tissue_positions_list.csv
# =========================================================

positions_candidates = [
    SPATIAL_DIR
    / "tissue_positions.csv",

    SPATIAL_DIR
    / "tissue_positions_list.csv",
]


POSITIONS_PATH = None


for p in positions_candidates:

    if p.exists():

        POSITIONS_PATH = p

        break


if POSITIONS_PATH is None:

    raise FileNotFoundError(
        "No tissue_positions.csv or "
        "tissue_positions_list.csv found."
    )


print(
    "\nPositions file:",
    POSITIONS_PATH
)


# =========================================================
# 3. Load filtered gene-expression matrix
# =========================================================

print(
    "\nLoading filtered 10x matrix..."
)


adata = sc.read_10x_h5(
    MATRIX_PATH
)


print(
    "Raw AnnData shape:",
    adata.shape
)


# Duplicate symbols can occur in 10x feature files.
adata.var_names_make_unique()


# Ensure string barcodes
adata.obs_names = (
    adata.obs_names
    .astype(str)
)


# =========================================================
# 4. Preserve counts
# =========================================================

adata.layers[
    "counts"
] = adata.X.copy()


# =========================================================
# 5. Load spatial coordinates
# =========================================================

print(
    "\nLoading spatial coordinates..."
)


if (
    POSITIONS_PATH.name
    ==
    "tissue_positions.csv"
):

    positions = pd.read_csv(
        POSITIONS_PATH
    )


    # Standardize possible column naming
    rename_map = {

        "pxl_row_in_fullres":
            "fullres_y",

        "pxl_col_in_fullres":
            "fullres_x",
    }


    positions = positions.rename(
        columns=rename_map
    )


else:

    # Older Space Ranger output has no header
    positions = pd.read_csv(
        POSITIONS_PATH,
        header=None,
        names=[
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "fullres_y",
            "fullres_x",
        ],
    )


# Defensive standardization
if "barcode" not in positions.columns:

    raise KeyError(
        "barcode column missing "
        "from tissue positions."
    )


positions[
    "barcode"
] = positions[
    "barcode"
].astype(str)


required_columns = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "fullres_y",
    "fullres_x",
]


for col in required_columns:

    if col not in positions.columns:

        raise KeyError(
            f"Missing column: {col}"
        )


positions = (
    positions
    .set_index(
        "barcode"
    )
)


# =========================================================
# 6. Match spatial coordinates to filtered barcodes
# =========================================================

missing_barcodes = [
    b
    for b in adata.obs_names
    if b not in positions.index
]


print(
    "Matrix barcodes missing spatial coordinates:",
    len(
        missing_barcodes
    )
)


if missing_barcodes:

    print(
        "Examples:",
        missing_barcodes[:10]
    )

    raise ValueError(
        "Spatial barcode mismatch."
    )


positions = positions.loc[
    adata.obs_names
].copy()


print(
    "Matched spots:",
    len(
        positions
    )
)


print(
    "in_tissue counts:"
)

print(
    positions[
        "in_tissue"
    ].value_counts(
        dropna=False
    )
)


# =========================================================
# 7. Add spatial metadata to AnnData.obs
# =========================================================

for col in [
    "in_tissue",
    "array_row",
    "array_col",
    "fullres_x",
    "fullres_y",
]:

    adata.obs[
        col
    ] = positions[
        col
    ].values


# =========================================================
# 8. Build obsm["spatial"]
#
# Standard convention:
#
# spatial[:, 0] = x = column coordinate
# spatial[:, 1] = y = row coordinate
# =========================================================

adata.obsm[
    "spatial"
] = np.column_stack(
    [
        adata.obs[
            "fullres_x"
        ].astype(float).values,

        adata.obs[
            "fullres_y"
        ].astype(float).values,
    ]
).astype(np.float64)


print(
    "\nSpatial coordinate range:"
)

print(
    "x:",
    adata.obsm[
        "spatial"
    ][
        :,
        0
    ].min(),
    "→",
    adata.obsm[
        "spatial"
    ][
        :,
        0
    ].max(),
)


print(
    "y:",
    adata.obsm[
        "spatial"
    ][
        :,
        1
    ].min(),
    "→",
    adata.obsm[
        "spatial"
    ][
        :,
        1
    ].max(),
)


# =========================================================
# 9. Load scale factors
# =========================================================

with open(
    SCALEFACTOR_PATH,
    "r"
) as f:

    scalefactors = json.load(
        f
    )


print(
    "\nScale factors:"
)


for key, value in scalefactors.items():

    print(
        key,
        "=",
        value
    )


# =========================================================
# 10. Load hires / lowres images
# =========================================================

hires_image = np.asarray(
    Image.open(
        HIRES_IMAGE_PATH
    ).convert("RGB")
)


lowres_image = np.asarray(
    Image.open(
        LOWRES_IMAGE_PATH
    ).convert("RGB")
)


print(
    "\nHires image:",
    hires_image.shape
)


print(
    "Lowres image:",
    lowres_image.shape
)


# =========================================================
# 11. Build Scanpy-compatible uns["spatial"]
# =========================================================

adata.uns[
    "spatial"
] = {

    LIBRARY_ID: {

        "images": {

            "hires":
                hires_image,

            "lowres":
                lowres_image,
        },

        "scalefactors":
            scalefactors,

        "metadata": {

            "sample_id":
                SAMPLE,

            "source":
                "10x Genomics Visium",

            "fullres_image_path":
                str(
                    FULLRES_IMAGE_PATH
                ),
        },
    }
}


# =========================================================
# 12. Normalize exactly as D9
#
# Preserve original counts in:
# layers["counts"]
#
# X becomes:
# normalize_total(1e4) + log1p
# =========================================================

print(
    "\nNormalizing expression..."
)


sc.pp.normalize_total(
    adata,
    target_sum=1e4,
)


sc.pp.log1p(
    adata
)


print(
    "Normalization complete."
)


# =========================================================
# 13. Target-gene QC
# =========================================================

gene_rows = []


print(
    "\nTarget gene QC:"
)


for gene in TARGET_GENES:

    if gene not in adata.var_names:

        raise KeyError(
            f"{gene} missing."
        )


    values = adata[
        :,
        gene
    ].X


    if hasattr(
        values,
        "toarray"
    ):

        values = values.toarray()


    values = (
        np.asarray(
            values
        )
        .ravel()
    )


    row = {

        "gene":
            gene,

        "mean":
            float(
                values.mean()
            ),

        "std":
            float(
                values.std()
            ),

        "min":
            float(
                values.min()
            ),

        "max":
            float(
                values.max()
            ),

        "nonzero_fraction":
            float(
                (
                    values > 0
                ).mean()
            ),
    }


    gene_rows.append(
        row
    )


    print(
        gene,
        "| mean=",
        round(
            row[
                "mean"
            ],
            4
        ),
        "| std=",
        round(
            row[
                "std"
            ],
            4
        ),
        "| nonzero=",
        round(
            row[
                "nonzero_fraction"
            ],
            4
        )
    )


gene_qc = pd.DataFrame(
    gene_rows
)


gene_qc.to_csv(
    OUTDIR
    / "external_target_gene_qc.csv",
    index=False,
)


# =========================================================
# 14. Build spatial graph
#
# Visium uses hexagonal grid geometry.
# =========================================================

print(
    "\nBuilding Squidpy spatial graph..."
)


sq.gr.spatial_neighbors(
    adata,
    coord_type="grid",
    n_neighs=6,
)


print(
    "Spatial graph:",
    adata.obsp[
        "spatial_connectivities"
    ].shape
)


print(
    "Spatial graph nnz:",
    adata.obsp[
        "spatial_connectivities"
    ].nnz
)


# =========================================================
# 15. Save external spot manifest
# =========================================================

manifest = pd.DataFrame(
    {
        "barcode":
            adata.obs_names,

        "in_tissue":
            adata.obs[
                "in_tissue"
            ].values,

        "array_row":
            adata.obs[
                "array_row"
            ].values,

        "array_col":
            adata.obs[
                "array_col"
            ].values,

        "fullres_x":
            adata.obs[
                "fullres_x"
            ].values,

        "fullres_y":
            adata.obs[
                "fullres_y"
            ].values,
    }
)


manifest.to_csv(
    OUTDIR
    / "external_spot_manifest.csv",
    index=False,
)


# =========================================================
# 16. Hires coordinate overlay QC
# =========================================================

print(
    "\nGenerating hires coordinate QC..."
)


hires_scale = float(
    scalefactors[
        "tissue_hires_scalef"
    ]
)


x_hires = (
    adata.obs[
        "fullres_x"
    ].astype(float).values
    *
    hires_scale
)


y_hires = (
    adata.obs[
        "fullres_y"
    ].astype(float).values
    *
    hires_scale
)


fig, ax = plt.subplots(
    figsize=(
        10,
        10
    )
)


ax.imshow(
    hires_image
)


ax.scatter(
    x_hires,
    y_hires,
    s=4,
    facecolors="none",
    edgecolors="red",
    linewidths=0.4,
)


ax.set_title(
    "D10 external Visium spot alignment\n"
    "V1_Mouse_Brain_Sagittal_Anterior"
)


ax.axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "01_external_hires_spot_overlay.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 17. Array-coordinate QC
# =========================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        7
    )
)


ax.scatter(
    adata.obs[
        "array_col"
    ],
    adata.obs[
        "array_row"
    ],
    s=5,
)


ax.invert_yaxis()


ax.set_aspect(
    "equal"
)


ax.set_xlabel(
    "Array column"
)


ax.set_ylabel(
    "Array row"
)


ax.set_title(
    "D10 Visium array geometry"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "02_external_visium_array_geometry.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 18. Full-resolution TIFF QC if available
# =========================================================

fullres_width = np.nan
fullres_height = np.nan


if FULLRES_IMAGE_PATH.exists():

    Image.MAX_IMAGE_PIXELS = None


    fullres = Image.open(
        FULLRES_IMAGE_PATH
    )


    fullres_width = (
        fullres.width
    )


    fullres_height = (
        fullres.height
    )


    print(
        "\nFull-resolution image:"
    )


    print(
        "Size:",
        fullres.size
    )


    print(
        "Mode:",
        fullres.mode
    )


    outside = (
        (
            adata.obs[
                "fullres_x"
            ].values
            <
            0
        )
        |
        (
            adata.obs[
                "fullres_x"
            ].values
            >=
            fullres.width
        )
        |
        (
            adata.obs[
                "fullres_y"
            ].values
            <
            0
        )
        |
        (
            adata.obs[
                "fullres_y"
            ].values
            >=
            fullres.height
        )
    )


    print(
        "Spot centers outside TIFF:",
        int(
            outside.sum()
        )
    )


# =========================================================
# 19. Dataset summary
# =========================================================

summary = pd.DataFrame(
    {
        "metric": [

            "sample_id",
            "n_spots",
            "n_genes",
            "hires_height",
            "hires_width",
            "lowres_height",
            "lowres_width",
            "tissue_hires_scalef",
            "tissue_lowres_scalef",
            "spot_diameter_fullres",
            "fullres_width",
            "fullres_height",
            "spatial_graph_nnz",
        ],

        "value": [

            SAMPLE,

            adata.n_obs,

            adata.n_vars,

            hires_image.shape[
                0
            ],

            hires_image.shape[
                1
            ],

            lowres_image.shape[
                0
            ],

            lowres_image.shape[
                1
            ],

            scalefactors.get(
                "tissue_hires_scalef",
                np.nan
            ),

            scalefactors.get(
                "tissue_lowres_scalef",
                np.nan
            ),

            scalefactors.get(
                "spot_diameter_fullres",
                np.nan
            ),

            fullres_width,

            fullres_height,

            adata.obsp[
                "spatial_connectivities"
            ].nnz,
        ],
    }
)


summary.to_csv(
    OUTDIR
    / "external_dataset_summary.csv",
    index=False,
)


# =========================================================
# 20. Save AnnData
# =========================================================

OUTPUT_H5AD = (
    OUTDIR
    / "V1_Mouse_Brain_Sagittal_Anterior_external.h5ad"
)


print(
    "\nSaving AnnData..."
)


adata.write_h5ad(
    OUTPUT_H5AD,
    compression="gzip",
)


# =========================================================
# Final report
# =========================================================

print(
    "\n======================================"
)

print(
    "D10 EXTERNAL DATASET READY"
)

print(
    "======================================"
)


print(
    "Shape:",
    adata.shape
)


print(
    "Spatial coordinates:",
    adata.obsm[
        "spatial"
    ].shape
)


print(
    "Graph:",
    adata.obsp[
        "spatial_connectivities"
    ].shape
)


print(
    "\nSaved:"
)


print(
    OUTPUT_H5AD
)


print(
    OUTDIR
    / "external_spot_manifest.csv"
)


print(
    OUTDIR
    / "external_target_gene_qc.csv"
)


print(
    OUTDIR
    / "external_dataset_summary.csv"
)


print(
    FIGDIR
    / "01_external_hires_spot_overlay.png"
)


print(
    FIGDIR
    / "02_external_visium_array_geometry.png"
)


print(
    "\nD10 external Visium preprocessing complete."
)
