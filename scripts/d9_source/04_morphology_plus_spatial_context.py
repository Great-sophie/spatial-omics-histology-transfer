import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)


# =========================================================
# Configuration
# =========================================================

# D8 AnnData containing spatial_connectivities
ADATA_PATH = Path(
    "/media/sophie/Data2T/Biomedical AI  skill/"
    "D8 Squidpy/outputs/visium_hne_spatial_graph.h5ad"
)

# D9 morphology features
MORPH_PCA_PATH = Path(
    "outputs/resnet50_image_embeddings_pca.npy"
)

MANIFEST_PATH = Path(
    "outputs/spot_image_embedding_manifest.csv"
)

# Reuse EXACT train/test split from script 03
SPLIT_PATH = Path(
    "outputs/prediction_split_manifest.csv"
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

RIDGE_ALPHA = 10.0


# =========================================================
# 1. Load morphology features
# =========================================================

print("Loading morphology PCA...")

X_center = np.load(
    MORPH_PCA_PATH
).astype(np.float32)

manifest = pd.read_csv(
    MANIFEST_PATH
)

split_df = pd.read_csv(
    SPLIT_PATH
)

print("Central morphology:", X_center.shape)
print("Manifest:", manifest.shape)
print("Split manifest:", split_df.shape)

assert X_center.shape[0] == len(manifest)

assert np.isnan(
    X_center
).sum() == 0


# =========================================================
# 2. Load spatial AnnData
# =========================================================

print("\nLoading spatial AnnData...")

adata = sc.read_h5ad(
    ADATA_PATH
)

print(adata)

print(
    "\nobsp keys:",
    list(adata.obsp.keys())
)

if (
    "spatial_connectivities"
    not in adata.obsp
):

    raise KeyError(
        "spatial_connectivities not found. "
        "Run D8 spatial graph analysis first."
    )


# =========================================================
# 3. Align everything by barcode
# =========================================================

barcodes = (
    manifest["barcode"]
    .astype(str)
    .tolist()
)


missing_adata = [
    b
    for b in barcodes
    if b not in adata.obs_names
]

if missing_adata:

    raise ValueError(
        f"{len(missing_adata)} barcodes "
        "missing from AnnData."
    )


adata = adata[
    barcodes
].copy()


# Align split manifest

split_df[
    "barcode"
] = split_df[
    "barcode"
].astype(str)


split_df = (
    split_df
    .set_index("barcode")
    .loc[barcodes]
    .reset_index()
)


assert list(
    adata.obs_names
) == barcodes


assert list(
    split_df["barcode"]
) == barcodes


print(
    "\nBarcode alignment successful:",
    len(barcodes),
    "spots"
)


# =========================================================
# 4. Spatial graph
# =========================================================

A = (
    adata.obsp[
        "spatial_connectivities"
    ]
    .tocsr()
    .astype(np.float32)
    .copy()
)


# Remove self-connections if present
A.setdiag(0)
A.eliminate_zeros()


degree = np.asarray(
    A.sum(axis=1)
).ravel()


print("\n=== Spatial graph ===")

print(
    "Shape:",
    A.shape
)

print(
    "Non-zero edges:",
    A.nnz
)

print(
    "Mean degree:",
    float(degree.mean())
)

print(
    "Median degree:",
    float(
        np.median(degree)
    )
)

print(
    "Min degree:",
    float(degree.min())
)

print(
    "Max degree:",
    float(degree.max())
)

print(
    "Zero-degree spots:",
    int(
        np.sum(
            degree == 0
        )
    )
)


# =========================================================
# 5. Row-normalized spatial adjacency
#
# W_ij = A_ij / degree_i
#
# X_neighbor = W @ X_center
#
# Each spot receives the mean morphology representation
# of its 1-hop spatial neighbors.
# =========================================================

safe_degree = degree.copy()

safe_degree[
    safe_degree == 0
] = 1.0


D_inv = sparse.diags(
    1.0 / safe_degree
)


W = (
    D_inv @ A
).tocsr()


print(
    "\nComputing 1-hop neighbor morphology..."
)


X_neighbor = (
    W @ X_center
).astype(np.float32)


print(
    "Neighbor morphology:",
    X_neighbor.shape
)


print(
    "Neighbor NaNs:",
    np.isnan(
        X_neighbor
    ).sum()
)


# =========================================================
# 6. Spatial-context representation
#
# Central morphology: 50-D
# Neighbor mean:      50-D
# -------------------------
# Combined:          100-D
# =========================================================

X_context = np.concatenate(
    [
        X_center,
        X_neighbor,
    ],
    axis=1,
).astype(np.float32)


print(
    "\nSpatial-context feature matrix:",
    X_context.shape
)


assert (
    X_context.shape[1]
    ==
    X_center.shape[1] * 2
)


assert (
    np.isnan(
        X_context
    ).sum()
    == 0
)


# =========================================================
# 7. Save spatial-context features
# =========================================================

np.save(
    OUTDIR
    / "resnet50_morphology_plus_spatial_context.npy",
    X_context,
)


np.save(
    OUTDIR
    / "resnet50_neighbor_morphology.npy",
    X_neighbor,
)


graph_summary = pd.DataFrame(
    {
        "metric": [
            "n_spots",
            "central_feature_dim",
            "neighbor_feature_dim",
            "context_feature_dim",
            "mean_degree",
            "median_degree",
            "min_degree",
            "max_degree",
            "zero_degree_spots",
        ],

        "value": [
            len(manifest),
            X_center.shape[1],
            X_neighbor.shape[1],
            X_context.shape[1],
            float(
                degree.mean()
            ),
            float(
                np.median(
                    degree
                )
            ),
            float(
                degree.min()
            ),
            float(
                degree.max()
            ),
            int(
                np.sum(
                    degree == 0
                )
            ),
        ],
    }
)


graph_summary.to_csv(
    OUTDIR
    / "spatial_context_feature_summary.csv",
    index=False,
)


# =========================================================
# 8. Extract normalized/log expression from adata.X
# =========================================================

print(
    "\nUsing adata.X "
    "(normalized/log-transformed expression)"
)


Y = {}


for gene in TARGET_GENES:

    if gene not in adata.var_names:

        raise KeyError(
            f"{gene} not found "
            "in adata.var_names."
        )


    values = (
        adata[
            :,
            gene
        ].X
    )


    if sparse.issparse(
        values
    ):

        values = (
            values
            .toarray()
        )


    values = (
        np.asarray(
            values
        )
        .ravel()
        .astype(
            np.float32
        )
    )


    Y[
        gene
    ] = values


    print(
        f"{gene}: "
        f"mean={values.mean():.4f}, "
        f"std={values.std():.4f}, "
        f"min={values.min():.4f}, "
        f"max={values.max():.4f}"
    )


# =========================================================
# 9. Recover fixed train/test splits
# =========================================================

random_train = np.where(
    split_df[
        "random_split"
    ].values
    == "train"
)[0]


random_test = np.where(
    split_df[
        "random_split"
    ].values
    == "test"
)[0]


spatial_train = np.where(
    split_df[
        "spatial_split"
    ].values
    == "train"
)[0]


spatial_test = np.where(
    split_df[
        "spatial_split"
    ].values
    == "test"
)[0]


print("\n=== Fixed splits ===")

print(
    "Random:",
    len(random_train),
    "train /",
    len(random_test),
    "test"
)

print(
    "Spatial:",
    len(spatial_train),
    "train /",
    len(spatial_test),
    "test"
)


splits = {

    "random": (
        random_train,
        random_test,
    ),

    "spatial_block": (
        spatial_train,
        spatial_test,
    ),
}


# =========================================================
# 10. Feature sets
#
# MorphologyOnly:
#     central 50-D
#
# NeighborOnly:
#     neighbor mean 50-D
#
# MorphologyPlusSpatial:
#     central + neighbor = 100-D
# =========================================================

feature_sets = {

    "MorphologyOnly":
        X_center,

    "NeighborOnly":
        X_neighbor,

    "MorphologyPlusSpatial":
        X_context,
}


# =========================================================
# 11. Ridge model
# =========================================================

def make_ridge():

    return Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),

            (
                "model",
                Ridge(
                    alpha=RIDGE_ALPHA
                )
            ),
        ]
    )


# =========================================================
# 12. Metrics
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


    if (
        np.std(
            y_true
        ) > 0
        and
        np.std(
            y_pred
        ) > 0
    ):

        pearson = pearsonr(
            y_true,
            y_pred
        ).statistic


        spearman = spearmanr(
            y_true,
            y_pred
        ).statistic

    else:

        pearson = np.nan
        spearman = np.nan


    return {

        "pearson_r":
            float(
                pearson
            ),

        "spearman_rho":
            float(
                spearman
            ),

        "r2":
            float(
                r2
            ),

        "rmse":
            float(
                rmse
            ),
    }


# =========================================================
# 13. Run experiments
# =========================================================

results = []

prediction_tables = {}


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


    prediction_table = pd.DataFrame(
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
                ],

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


        for (
            feature_name,
            X
        ) in feature_sets.items():


            model = make_ridge()


            model.fit(
                X[
                    train_idx
                ],
                y[
                    train_idx
                ],
            )


            pred = model.predict(
                X[
                    test_idx
                ]
            )


            metric = evaluate(
                y[
                    test_idx
                ],
                pred,
            )


            results.append(
                {

                    "gene":
                        gene,

                    "split":
                        split_name,

                    "feature_set":
                        feature_name,

                    "model":
                        "Ridge",

                    "n_train":
                        len(
                            train_idx
                        ),

                    "n_test":
                        len(
                            test_idx
                        ),

                    "feature_dim":
                        X.shape[1],

                    **metric,
                }
            )


            print(
                f"{feature_name:22s}"
                f" | r={metric['pearson_r']:.3f}"
                f" | rho={metric['spearman_rho']:.3f}"
                f" | R2={metric['r2']:.3f}"
                f" | RMSE={metric['rmse']:.3f}"
            )


            pred_col = (
                f"{split_name}_"
                f"{feature_name}_pred"
            )


            prediction_table[
                pred_col
            ] = np.nan


            prediction_table.loc[
                test_idx,
                pred_col
            ] = pred


    prediction_tables[
        gene
    ] = prediction_table


    prediction_table.to_csv(
        OUTDIR
        / (
            f"spatial_context_prediction_"
            f"{gene}.csv"
        ),
        index=False,
    )


# =========================================================
# 14. Save metrics
# =========================================================

metrics = pd.DataFrame(
    results
)


metrics.to_csv(
    OUTDIR
    / "morphology_spatial_context_metrics.csv",
    index=False,
)


print(
    "\n=== All metrics ==="
)


print(
    metrics
    .sort_values(
        [
            "gene",
            "split",
            "feature_set",
        ]
    )
    .to_string(
        index=False
    )
)


# =========================================================
# 15. Calculate improvement over morphology-only
# =========================================================

improvement_rows = []


for gene in TARGET_GENES:

    for split_name in [
        "random",
        "spatial_block",
    ]:


        sub = metrics[
            (
                metrics[
                    "gene"
                ]
                == gene
            )
            &
            (
                metrics[
                    "split"
                ]
                == split_name
            )
        ]


        base = sub[
            sub[
                "feature_set"
            ]
            ==
            "MorphologyOnly"
        ].iloc[0]


        context = sub[
            sub[
                "feature_set"
            ]
            ==
            "MorphologyPlusSpatial"
        ].iloc[0]


        improvement_rows.append(
            {

                "gene":
                    gene,

                "split":
                    split_name,

                "morphology_pearson":
                    base[
                        "pearson_r"
                    ],

                "context_pearson":
                    context[
                        "pearson_r"
                    ],

                "delta_pearson":
                    (
                        context[
                            "pearson_r"
                        ]
                        -
                        base[
                            "pearson_r"
                        ]
                    ),

                "morphology_r2":
                    base[
                        "r2"
                    ],

                "context_r2":
                    context[
                        "r2"
                    ],

                "delta_r2":
                    (
                        context[
                            "r2"
                        ]
                        -
                        base[
                            "r2"
                        ]
                    ),

                "morphology_rmse":
                    base[
                        "rmse"
                    ],

                "context_rmse":
                    context[
                        "rmse"
                    ],

                "delta_rmse":
                    (
                        context[
                            "rmse"
                        ]
                        -
                        base[
                            "rmse"
                        ]
                    ),
            }
        )


improvement = pd.DataFrame(
    improvement_rows
)


improvement.to_csv(
    OUTDIR
    / "spatial_context_improvement.csv",
    index=False,
)


print(
    "\n=== Spatial-context improvement ==="
)


print(
    improvement
    .to_string(
        index=False
    )
)


# =========================================================
# 16. Comparison plot
#
# Spatial-block Pearson r
# =========================================================

plot_df = metrics[
    metrics[
        "split"
    ]
    ==
    "spatial_block"
].copy()


pivot = plot_df.pivot(
    index="gene",
    columns="feature_set",
    values="pearson_r",
)


gene_order = [
    gene
    for gene in TARGET_GENES
    if gene in pivot.index
]


pivot = pivot.loc[
    gene_order
]


x_pos = np.arange(
    len(
        pivot
    )
)

width = 0.25


fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.bar(
    x_pos
    - width,
    pivot[
        "MorphologyOnly"
    ].values,
    width,
    label="Morphology only",
)


ax.bar(
    x_pos,
    pivot[
        "NeighborOnly"
    ].values,
    width,
    label="Neighbor morphology",
)


ax.bar(
    x_pos
    + width,
    pivot[
        "MorphologyPlusSpatial"
    ].values,
    width,
    label="Morphology + spatial context",
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
    "Spatial-block generalization:\n"
    "central vs neighbor morphology"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "17_spatial_context_model_comparison.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 17. Delta Pearson plot
# =========================================================

delta_plot = improvement[
    improvement[
        "split"
    ]
    ==
    "spatial_block"
].copy()


fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(
    delta_plot[
        "gene"
    ],
    delta_plot[
        "delta_pearson"
    ],
)


ax.axhline(
    0,
    linewidth=1,
)


ax.set_ylabel(
    "Δ Pearson r"
)


ax.set_title(
    "Benefit of adding 1-hop spatial morphology context"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "18_spatial_context_delta_pearson.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 18. Spatial prediction maps
#
# Morphology + spatial context
# Spatial-block test only
# =========================================================

figure_number = 19


for gene in TARGET_GENES:


    df = prediction_tables[
        gene
    ]


    pred_col = (
        "spatial_block_"
        "MorphologyPlusSpatial_pred"
    )


    valid = df[
        pred_col
    ].notna()


    x = df.loc[
        valid,
        "fullres_x"
    ].values


    y_coord = df.loc[
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


    error_lim = max(
        abs(
            error.min()
        ),
        abs(
            error.max()
        ),
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )


    s0 = axes[
        0
    ].scatter(
        x,
        y_coord,
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


    axes[
        1
    ].scatter(
        x,
        y_coord,
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


    s2 = axes[
        2
    ].scatter(
        x,
        y_coord,
        c=error,
        s=14,
        vmin=-error_lim,
        vmax=error_lim,
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
        label=(
            "Normalized/log expression"
        ),
    )


    fig.colorbar(
        s2,
        ax=axes[2],
        shrink=0.8,
        label=(
            "Predicted - true"
        ),
    )


    fig.suptitle(
        f"{gene} | Morphology + "
        "1-hop spatial context"
    )


    plt.savefig(
        FIGDIR
        / (
            f"{figure_number:02d}_"
            f"{gene}_"
            "spatial_context_prediction.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


    figure_number += 1


# =========================================================
# 19. Final report
# =========================================================

print("\nSaved:")

print(
    "outputs/resnet50_neighbor_morphology.npy"
)

print(
    "outputs/resnet50_morphology_plus_spatial_context.npy"
)

print(
    "outputs/spatial_context_feature_summary.csv"
)

print(
    "outputs/morphology_spatial_context_metrics.csv"
)

print(
    "outputs/spatial_context_improvement.csv"
)

print(
    "outputs/spatial_context_prediction_<gene>.csv"
)

print(
    "figures/17_spatial_context_model_comparison.png"
)

print(
    "figures/18_spatial_context_delta_pearson.png"
)

print(
    "figures/19-22 spatial context prediction maps"
)


print(
    "\nD9 morphology + spatial-context experiment complete."
)
