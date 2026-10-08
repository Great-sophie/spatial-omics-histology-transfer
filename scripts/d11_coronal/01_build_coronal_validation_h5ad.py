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
    "D11_mouse_brain_coronal_validation"
)

DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


SAMPLE = "Visium_Fresh_Frozen_Adult_Mouse_Brain"


MATRIX_PATH = (
    DATA
    / f"{SAMPLE}_filtered_feature_bc_matrix.h5"
)

FULLRES_IMAGE_PATH = (
    DATA
    / f"{SAMPLE}_image.tif"
)

SPATIAL_DIR = (
    DATA
    / "spatial"
)

POSITIONS_PATH = (
    SPATIAL_DIR
    / "tissue_positions.csv"
)

SCALE_PATH = (
    SPATIAL_DIR
    / "scalefactors_json.json"
)

HIRES_PATH = (
    SPATIAL_DIR
    / "tissue_hires_image.png"
)

LOWRES_PATH = (
    SPATIAL_DIR
    / "tissue_lowres_image.png"
)


TARGET_GENES = [
    "Camk2n1",
    "Slc17a7",
    "Plp1",
    "Cldn11",
]


# =========================================================
# Input checks
# =========================================================

print("Checking input files...")


required = [
    MATRIX_PATH,
    FULLRES_IMAGE_PATH,
    POSITIONS_PATH,
    SCALE_PATH,
    HIRES_PATH,
    LOWRES_PATH,
]


for p in required:

    print(
        p,
        "FOUND" if p.exists() else "MISSING"
    )

    if not p.exists():

        raise FileNotFoundError(
            str(p)
        )


# =========================================================
# Load matrix
# =========================================================

print(
    "\nLoading 10x matrix..."
)


adata = sc.read_10x_h5(
    MATRIX_PATH
)


print(
    "Raw shape:",
    adata.shape
)


adata.var_names_make_unique()

adata.obs_names = (
    adata.obs_names
    .astype(str)
)


# preserve raw counts
adata.layers[
    "counts"
] = adata.X.copy()


# =========================================================
# Target gene presence
# =========================================================

print(
    "\nTarget genes:"
)


for gene in TARGET_GENES:

    present = (
        gene
        in
        adata.var_names
    )

    print(
        gene,
        present
    )

    if not present:

        raise KeyError(
            f"{gene} missing"
        )


# =========================================================
# Spatial positions
# =========================================================

print(
    "\nLoading spatial positions..."
)


positions = pd.read_csv(
    POSITIONS_PATH
)


print(
    "Columns:",
    list(
        positions.columns
    )
)


rename_map = {
    "pxl_row_in_fullres":
        "fullres_y",

    "pxl_col_in_fullres":
        "fullres_x",
}


positions = positions.rename(
    columns=rename_map
)


required_cols = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "fullres_y",
    "fullres_x",
]


for col in required_cols:

    if col not in positions.columns:

        raise KeyError(
            f"Missing column: {col}"
        )


positions[
    "barcode"
] = (
    positions[
        "barcode"
    ]
    .astype(str)
)


positions = positions.set_index(
    "barcode"
)


# =========================================================
# Match filtered barcodes
# =========================================================

missing = [
    b
    for b in adata.obs_names
    if b not in positions.index
]


print(
    "Missing spatial barcodes:",
    len(missing)
)


if missing:

    raise RuntimeError(
        "Matrix/spatial barcode mismatch."
    )


positions = positions.loc[
    adata.obs_names
].copy()


print(
    "Matched spots:",
    len(positions)
)


print(
    "in_tissue:"
)

print(
    positions[
        "in_tissue"
    ].value_counts()
)


# =========================================================
# Add obs metadata
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
# obsm["spatial"]
#
# x, y
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
    "\nCoordinate ranges:"
)

print(
    "x:",
    adata.obsm[
        "spatial"
    ][
        :,
        0
    ].min(),
    "->",
    adata.obsm[
        "spatial"
    ][
        :,
        0
    ].max()
)

print(
    "y:",
    adata.obsm[
        "spatial"
    ][
        :,
        1
    ].min(),
    "->",
    adata.obsm[
        "spatial"
    ][
        :,
        1
    ].max()
)


# =========================================================
# Scale factors
# =========================================================

with open(
    SCALE_PATH
) as f:

    scales = json.load(f)


print(
    "\nScale factors:"
)


for k, v in scales.items():

    print(
        k,
        "=",
        v
    )


# =========================================================
# Images
# =========================================================

hires = np.asarray(
    Image.open(
        HIRES_PATH
    ).convert("RGB")
)


lowres = np.asarray(
    Image.open(
        LOWRES_PATH
    ).convert("RGB")
)


Image.MAX_IMAGE_PIXELS = None

full = Image.open(
    FULLRES_IMAGE_PATH
)


print(
    "\nHires:",
    hires.shape
)

print(
    "Lowres:",
    lowres.shape
)

print(
    "Fullres:",
    full.size
)


# =========================================================
# Coordinate bounds QC
# =========================================================

x = (
    adata.obs[
        "fullres_x"
    ]
    .astype(float)
    .values
)

y = (
    adata.obs[
        "fullres_y"
    ]
    .astype(float)
    .values
)


outside = (
    (x < 0)
    |
    (x >= full.width)
    |
    (y < 0)
    |
    (y >= full.height)
)


print(
    "Spot centers outside TIFF:",
    int(
        outside.sum()
    )
)


if outside.any():

    raise RuntimeError(
        "Spatial coordinates do not align "
        "with full-resolution TIFF."
    )


# =========================================================
# Build uns["spatial"]
# =========================================================

adata.uns[
    "spatial"
] = {
    SAMPLE: {

        "images": {
            "hires":
                hires,

            "lowres":
                lowres,
        },

        "scalefactors":
            scales,

        "metadata": {
            "sample_id":
                SAMPLE,

            "validation_role":
                "independent_coronal_slide",

            "fullres_image_path":
                str(
                    FULLRES_IMAGE_PATH
                ),
        },
    }
}


# =========================================================
# Normalize exactly like D9/D10
# =========================================================

print(
    "\nNormalizing..."
)


sc.pp.normalize_total(
    adata,
    target_sum=1e4
)


sc.pp.log1p(
    adata
)


# =========================================================
# Target gene QC
# =========================================================

gene_rows = []


print(
    "\nTarget gene expression QC:"
)


for gene in TARGET_GENES:

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


pd.DataFrame(
    gene_rows
).to_csv(
    OUT
    / "coronal_target_gene_qc.csv",
    index=False
)


# =========================================================
# Spatial graph
# =========================================================

print(
    "\nBuilding spatial graph..."
)


sq.gr.spatial_neighbors(
    adata,
    coord_type="grid",
    n_neighs=6
)


print(
    "Graph:",
    adata.obsp[
        "spatial_connectivities"
    ].shape
)

print(
    "Graph nnz:",
    adata.obsp[
        "spatial_connectivities"
    ].nnz
)


# =========================================================
# Save manifest
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
    OUT
    / "coronal_spot_manifest.csv",
    index=False
)


# =========================================================
# Hires spot overlay
# =========================================================

hires_scale = float(
    scales[
        "tissue_hires_scalef"
    ]
)


xh = (
    x
    *
    hires_scale
)

yh = (
    y
    *
    hires_scale
)


fig, ax = plt.subplots(
    figsize=(10, 10)
)


ax.imshow(
    hires
)


ax.scatter(
    xh,
    yh,
    s=4,
    facecolors="none",
    edgecolors="red",
    linewidths=0.4
)


ax.set_title(
    "D11 coronal validation\n"
    "Visium_Fresh_Frozen_Adult_Mouse_Brain"
)


ax.axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    FIG
    / "01_coronal_hires_spot_overlay.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# =========================================================
# Array geometry
# =========================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)


ax.scatter(
    adata.obs[
        "array_col"
    ],
    adata.obs[
        "array_row"
    ],
    s=5
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
    "D11 Visium array geometry"
)


plt.tight_layout()


plt.savefig(
    FIG
    / "02_coronal_visium_array_geometry.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# =========================================================
# Summary
# =========================================================

summary = pd.DataFrame(
    {
        "metric": [
            "sample_id",
            "n_spots",
            "n_genes",
            "spot_diameter_fullres",
            "tissue_hires_scalef",
            "tissue_lowres_scalef",
            "hires_width",
            "hires_height",
            "fullres_width",
            "fullres_height",
            "spatial_graph_nnz",
        ],

        "value": [
            SAMPLE,
            adata.n_obs,
            adata.n_vars,

            scales.get(
                "spot_diameter_fullres",
                np.nan
            ),

            scales.get(
                "tissue_hires_scalef",
                np.nan
            ),

            scales.get(
                "tissue_lowres_scalef",
                np.nan
            ),

            hires.shape[
                1
            ],

            hires.shape[
                0
            ],

            full.width,
            full.height,

            adata.obsp[
                "spatial_connectivities"
            ].nnz,
        ]
    }
)


summary.to_csv(
    OUT
    / "coronal_dataset_summary.csv",
    index=False
)


# =========================================================
# Save h5ad
# =========================================================

H5AD_OUT = (
    OUT
    / "Visium_Fresh_Frozen_Adult_Mouse_Brain_coronal.h5ad"
)


print(
    "\nSaving..."
)


adata.write_h5ad(
    H5AD_OUT,
    compression="gzip"
)


# =========================================================
# Final
# =========================================================

print(
    "\n======================================"
)

print(
    "D11 CORONAL DATASET READY"
)

print(
    "======================================"
)


print(
    "Shape:",
    adata.shape
)

print(
    "Spatial:",
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
    H5AD_OUT
)

print(
    OUT
    / "coronal_spot_manifest.csv"
)

print(
    OUT
    / "coronal_target_gene_qc.csv"
)

print(
    OUT
    / "coronal_dataset_summary.csv"
)

print(
    FIG
    / "01_coronal_hires_spot_overlay.png"
)

print(
    FIG
    / "02_coronal_visium_array_geometry.png"
)
