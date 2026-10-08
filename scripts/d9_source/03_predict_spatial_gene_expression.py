import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)


# =========================================================
# Configuration
# =========================================================

ADATA_PATH = Path(
    "/media/sophie/Data2T/Biomedical AI  skill/"
    "D8 Squidpy/data/anndata/visium_hne_adata.h5ad"
)

PCA_PATH = Path(
    "outputs/resnet50_image_embeddings_pca.npy"
)

MANIFEST_PATH = Path(
    "outputs/spot_image_embedding_manifest.csv"
)

OUTDIR = Path("outputs")
FIGDIR = Path("figures")

OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)


TARGET_GENES = [
    "Camk2n1",
    "Slc17a7",
    "Plp1",
    "Cldn11",
]

RANDOM_STATE = 42

TEST_SIZE = 0.20


# =========================================================
# 1. Load morphology features
# =========================================================

print("Loading morphology embeddings...")

X_morph = np.load(PCA_PATH)

manifest = pd.read_csv(
    MANIFEST_PATH
)

print(
    "Morphology PCA:",
    X_morph.shape
)

print(
    "Manifest:",
    manifest.shape
)

assert (
    X_morph.shape[0]
    == len(manifest)
)

assert (
    np.isnan(X_morph).sum()
    == 0
)


# =========================================================
# 2. Load Visium expression
# =========================================================

print(
    "\nLoading Visium expression..."
)

adata = sc.read_h5ad(
    ADATA_PATH
)

print(adata)

print(
    "\nExpression matrix shape:",
    adata.X.shape
)

print(
    "Using adata.X "
    "(normalized/log-transformed expression)"
)


# =========================================================
# 3. Align morphology and molecular data by barcode
# =========================================================

barcodes = (
    manifest["barcode"]
    .astype(str)
    .tolist()
)

missing = [
    barcode
    for barcode in barcodes
    if barcode not in adata.obs_names
]

if missing:

    raise ValueError(
        f"{len(missing)} barcodes "
        "missing from AnnData."
    )


adata_aligned = adata[
    barcodes
].copy()


assert (
    list(
        adata_aligned.obs_names
    )
    == barcodes
)


print(
    "\nBarcode alignment successful:",
    adata_aligned.n_obs,
    "spots"
)


# =========================================================
# 4. Extract normalized/log-transformed target expression
#
# IMPORTANT:
# Use adata.X, NOT adata.raw.
#
# This keeps D9 prediction targets consistent with
# the normalized/log-transformed representation used
# in D8 spatial analysis.
# =========================================================

Y = {}

print(
    "\nTarget gene expression:"
)


for gene in TARGET_GENES:

    if gene not in adata_aligned.var_names:

        raise KeyError(
            f"{gene} not found "
            "in adata.var_names."
        )


    values = (
        adata_aligned[
            :,
            gene
        ].X
    )


    if sparse.issparse(values):

        values = (
            values
            .toarray()
        )


    values = (
        np.asarray(values)
        .ravel()
        .astype(np.float32)
    )


    Y[gene] = values


    print(
        f"{gene}: "
        f"mean={values.mean():.4f}, "
        f"std={values.std():.4f}, "
        f"min={values.min():.4f}, "
        f"max={values.max():.4f}, "
        f"zero_fraction="
        f"{np.mean(values == 0):.4f}"
    )


# =========================================================
# 5. Random train/test split
#
# Teaching baseline only.
# Nearby spots may appear in both train and test,
# creating spatial leakage.
# =========================================================

indices = np.arange(
    len(manifest)
)


train_random, test_random = (
    train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
)


print(
    "\nRandom split:"
)

print(
    len(train_random),
    "train /",
    len(test_random),
    "test"
)


# =========================================================
# 6. Construct spatial blocks
#
# Tissue is divided into a 4 × 4 quantile grid.
# Entire spatial blocks are assigned to either
# train or test.
# =========================================================

print(
    "\nConstructing spatial blocks..."
)


x_coord = (
    manifest[
        "fullres_x"
    ]
    .astype(float)
)


y_coord = (
    manifest[
        "fullres_y"
    ]
    .astype(float)
)


x_bin = pd.qcut(
    x_coord,
    q=4,
    labels=False,
    duplicates="drop",
)


y_bin = pd.qcut(
    y_coord,
    q=4,
    labels=False,
    duplicates="drop",
)


spatial_group = (
    x_bin.astype(str)
    + "_"
    + y_bin.astype(str)
)


manifest[
    "spatial_block"
] = spatial_group


print(
    "Number of spatial blocks:",
    manifest[
        "spatial_block"
    ].nunique()
)


# =========================================================
# 7. Spatial whole-block holdout
# =========================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=RANDOM_STATE,
)


train_spatial, test_spatial = next(
    gss.split(
        X_morph,
        groups=manifest[
            "spatial_block"
        ],
    )
)


print(
    "\nSpatial block split:"
)

print(
    len(train_spatial),
    "train /",
    len(test_spatial),
    "test"
)


train_blocks = sorted(
    manifest.iloc[
        train_spatial
    ][
        "spatial_block"
    ].unique()
)


test_blocks = sorted(
    manifest.iloc[
        test_spatial
    ][
        "spatial_block"
    ].unique()
)


print(
    "\nSpatial train blocks:"
)

print(
    train_blocks
)


print(
    "\nSpatial test blocks:"
)

print(
    test_blocks
)


# =========================================================
# 8. Save split manifest
# =========================================================

split_columns = [
    "barcode",
    "fullres_x",
    "fullres_y",
    "cluster",
    "spatial_block",
]

split_columns = [
    col
    for col in split_columns
    if col in manifest.columns
]


split_df = manifest[
    split_columns
].copy()


split_df[
    "random_split"
] = "train"

split_df.loc[
    test_random,
    "random_split"
] = "test"


split_df[
    "spatial_split"
] = "train"

split_df.loc[
    test_spatial,
    "spatial_split"
] = "test"


split_df.to_csv(
    OUTDIR
    / "prediction_split_manifest.csv",
    index=False,
)


# =========================================================
# 9. Plot split QC
# =========================================================

def plot_split(
    split_column,
    filename,
    title,
):

    is_test = (
        split_df[
            split_column
        ].values
        == "test"
    )


    fig, ax = plt.subplots(
        figsize=(8, 8)
    )


    scatter = ax.scatter(
        split_df[
            "fullres_x"
        ],
        split_df[
            "fullres_y"
        ],
        c=is_test.astype(int),
        s=12,
    )


    ax.invert_yaxis()

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "Spatial x"
    )

    ax.set_ylabel(
        "Spatial y"
    )

    ax.set_title(
        title
    )


    cbar = plt.colorbar(
        scatter,
        ax=ax,
    )

    cbar.set_ticks(
        [0, 1]
    )

    cbar.set_ticklabels(
        [
            "Train",
            "Test",
        ]
    )


    plt.tight_layout()


    plt.savefig(
        FIGDIR
        / filename,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


plot_split(
    "random_split",
    "06_random_train_test_split.png",
    "Random train/test split",
)


plot_split(
    "spatial_split",
    "07_spatial_block_train_test_split.png",
    "Spatial block train/test split",
)


# =========================================================
# 10. Models
# =========================================================

models = {

    "Ridge": Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),

            (
                "model",
                Ridge(
                    alpha=10.0
                )
            ),
        ]
    ),


    "RandomForest":
        RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
}


# =========================================================
# 11. Evaluation metrics
# =========================================================

def evaluate(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )


    r2 = r2_score(
        y_true,
        y_pred,
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


    # Pearson correlation
    if (
        np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):

        pearson = pearsonr(
            y_true,
            y_pred,
        ).statistic

    else:

        pearson = np.nan


    # Spearman correlation
    if (
        np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):

        spearman = spearmanr(
            y_true,
            y_pred,
        ).statistic

    else:

        spearman = np.nan


    return {

        "pearson_r":
            float(pearson)
            if not np.isnan(pearson)
            else np.nan,

        "spearman_rho":
            float(spearman)
            if not np.isnan(spearman)
            else np.nan,

        "r2":
            float(r2),

        "rmse":
            float(rmse),
    }


# =========================================================
# 12. Prediction experiments
# =========================================================

metrics_rows = []

all_prediction_tables = {}


splits = {

    "random": (
        train_random,
        test_random,
    ),

    "spatial_block": (
        train_spatial,
        test_spatial,
    ),
}


for gene in TARGET_GENES:

    print(
        "\n======================================"
    )

    print(
        "Gene:",
        gene
    )

    print(
        "======================================"
    )


    y = Y[
        gene
    ]


    gene_prediction_table = pd.DataFrame(
        {

            "barcode":
                manifest[
                    "barcode"
                ],

            "fullres_x":
                manifest[
                    "fullres_x"
                ],

            "fullres_y":
                manifest[
                    "fullres_y"
                ],

            "cluster":
                manifest[
                    "cluster"
                ]
                if "cluster"
                in manifest.columns
                else "NA",

            "expression_true":
                y,
        }
    )


    for split_name, (
        train_idx,
        test_idx,
    ) in splits.items():


        print(
            "\nSplit:",
            split_name
        )


        X_train = X_morph[
            train_idx
        ]

        X_test = X_morph[
            test_idx
        ]


        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]


        # =================================================
        # Mean baseline
        # =================================================

        mean_pred = np.full(
            len(test_idx),
            y_train.mean(),
            dtype=float,
        )


        baseline_result = evaluate(
            y_test,
            mean_pred,
        )


        metrics_rows.append(
            {

                "gene":
                    gene,

                "split":
                    split_name,

                "model":
                    "MeanBaseline",

                "n_train":
                    len(train_idx),

                "n_test":
                    len(test_idx),

                **baseline_result,
            }
        )


        # =================================================
        # Train ML models
        # =================================================

        for (
            model_name,
            model
        ) in models.items():


            model.fit(
                X_train,
                y_train,
            )


            pred = model.predict(
                X_test
            )


            result = evaluate(
                y_test,
                pred,
            )


            metrics_rows.append(
                {

                    "gene":
                        gene,

                    "split":
                        split_name,

                    "model":
                        model_name,

                    "n_train":
                        len(train_idx),

                    "n_test":
                        len(test_idx),

                    **result,
                }
            )


            print(
                f"{model_name:12s}"
                f" | Pearson={result['pearson_r']:.3f}"
                f" | Spearman={result['spearman_rho']:.3f}"
                f" | R2={result['r2']:.3f}"
                f" | RMSE={result['rmse']:.3f}"
            )


            pred_col = (
                f"{split_name}_"
                f"{model_name}_pred"
            )


            gene_prediction_table[
                pred_col
            ] = np.nan


            gene_prediction_table.loc[
                test_idx,
                pred_col
            ] = pred


    all_prediction_tables[
        gene
    ] = gene_prediction_table


    gene_prediction_table.to_csv(
        OUTDIR
        / f"prediction_{gene}.csv",
        index=False,
    )


# =========================================================
# 13. Save full metrics
# =========================================================

metrics = pd.DataFrame(
    metrics_rows
)


metrics.to_csv(
    OUTDIR
    / "morphology_gene_prediction_metrics.csv",
    index=False,
)


print(
    "\n=== Prediction metrics ==="
)


print(
    metrics[
        metrics[
            "model"
        ]
        != "MeanBaseline"
    ]
    .sort_values(
        [
            "gene",
            "split",
            "model",
        ]
    )
    .to_string(
        index=False
    )
)


# =========================================================
# 14. Compact summary
# =========================================================

summary = metrics[
    metrics[
        "model"
    ].isin(
        [
            "Ridge",
            "RandomForest",
        ]
    )
].copy()


summary = summary[
    [
        "gene",
        "split",
        "model",
        "pearson_r",
        "spearman_rho",
        "r2",
        "rmse",
    ]
]


summary.to_csv(
    OUTDIR
    / "morphology_gene_prediction_summary.csv",
    index=False,
)


# =========================================================
# 15. Spatial prediction maps
#
# Ridge only for clean visualization.
# =========================================================

figure_number = 8


for gene in TARGET_GENES:

    df = all_prediction_tables[
        gene
    ]


    for split_name in [
        "random",
        "spatial_block",
    ]:


        pred_col = (
            f"{split_name}_"
            "Ridge_pred"
        )


        valid = df[
            pred_col
        ].notna()


        if valid.sum() == 0:
            continue


        x = df.loc[
            valid,
            "fullres_x"
        ].values


        y_coord_plot = df.loc[
            valid,
            "fullres_y"
        ].values


        true = df.loc[
            valid,
            "expression_true"
        ].values


        pred = df.loc[
            valid,
            pred_col
        ].values


        error = (
            pred
            - true
        )


        vmin = min(
            true.min(),
            pred.min(),
        )


        vmax = max(
            true.max(),
            pred.max(),
        )


        abs_error_max = max(
            abs(error.min()),
            abs(error.max()),
        )


        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5),
        )


        # True expression
        s0 = axes[
            0
        ].scatter(
            x,
            y_coord_plot,
            c=true,
            s=14,
            vmin=vmin,
            vmax=vmax,
        )


        axes[
            0
        ].invert_yaxis()

        axes[
            0
        ].set_aspect(
            "equal"
        )

        axes[
            0
        ].set_title(
            f"{gene}: true"
        )


        # Predicted expression
        axes[
            1
        ].scatter(
            x,
            y_coord_plot,
            c=pred,
            s=14,
            vmin=vmin,
            vmax=vmax,
        )


        axes[
            1
        ].invert_yaxis()

        axes[
            1
        ].set_aspect(
            "equal"
        )

        axes[
            1
        ].set_title(
            f"{gene}: predicted"
        )


        # Prediction error
        s2 = axes[
            2
        ].scatter(
            x,
            y_coord_plot,
            c=error,
            s=14,
            vmin=-abs_error_max,
            vmax=abs_error_max,
        )


        axes[
            2
        ].invert_yaxis()

        axes[
            2
        ].set_aspect(
            "equal"
        )

        axes[
            2
        ].set_title(
            "Prediction error"
        )


        fig.colorbar(
            s0,
            ax=axes[:2],
            shrink=0.8,
            label="Normalized/log expression",
        )


        fig.colorbar(
            s2,
            ax=axes[2],
            shrink=0.8,
            label="Predicted - true",
        )


        fig.suptitle(
            f"{gene} | Ridge | {split_name}"
        )


        plt.savefig(
            FIGDIR
            / (
                f"{figure_number:02d}_"
                f"{gene}_"
                f"{split_name}_"
                "ridge_spatial_prediction.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )


        plt.close()


        figure_number += 1


# =========================================================
# 16. Random vs spatial comparison plot
#
# Ridge Pearson r
# =========================================================

ridge_summary = summary[
    summary[
        "model"
    ]
    == "Ridge"
].copy()


pivot = ridge_summary.pivot(
    index="gene",
    columns="split",
    values="pearson_r",
)


genes_order = [
    gene
    for gene in TARGET_GENES
    if gene in pivot.index
]


pivot = pivot.loc[
    genes_order
]


x_pos = np.arange(
    len(pivot)
)

width = 0.35


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.bar(
    x_pos - width / 2,
    pivot[
        "random"
    ].values,
    width,
    label="Random split",
)


ax.bar(
    x_pos + width / 2,
    pivot[
        "spatial_block"
    ].values,
    width,
    label="Spatial block split",
)


ax.set_xticks(
    x_pos
)

ax.set_xticklabels(
    pivot.index
)

ax.set_ylabel(
    "Pearson r"
)

ax.set_title(
    "Morphology → gene expression prediction"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "16_random_vs_spatial_prediction.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 17. Final output report
# =========================================================

print(
    "\nSaved:"
)

print(
    "outputs/prediction_split_manifest.csv"
)

print(
    "outputs/morphology_gene_prediction_metrics.csv"
)

print(
    "outputs/morphology_gene_prediction_summary.csv"
)

print(
    "outputs/prediction_<gene>.csv"
)

print(
    "figures/06_random_train_test_split.png"
)

print(
    "figures/07_spatial_block_train_test_split.png"
)

print(
    "figures/08-15 gene spatial prediction maps"
)

print(
    "figures/16_random_vs_spatial_prediction.png"
)


print(
    "\nD9 morphology → molecular prediction complete."
)
