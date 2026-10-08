import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# =========================================================
# Configuration
# =========================================================

ADATA_PATH = Path(
    "/media/sophie/Data2T/Biomedical AI  skill/"
    "D8 Squidpy/outputs/visium_hne_spatial_graph.h5ad"
)

MANIFEST_PATH = Path(
    "outputs/spot_image_embedding_manifest.csv"
)

SPLIT_PATH = Path(
    "outputs/prediction_split_manifest.csv"
)

RESOLUTIONS = {

    "LowRes_38pxUpscaled": Path(
        "outputs/dinov2_vits14_image_embeddings.npy"
    ),

    "FullRes_224px": Path(
        "outputs/dinov2_vits14_fullres_image_embeddings.npy"
    ),
}


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


PCA_DIM = 50

HIDDEN_DIM = 64

DROPOUT = 0.20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 500

PATIENCE = 60

RIDGE_ALPHA = 10.0


SEEDS = [
    42,
    43,
    44,
    45,
    46,
]


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
# Reproducibility
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Metrics
# =========================================================

def evaluate_gene(
    y_true,
    y_pred,
):

    if (
        np.std(y_true) > 0
        and
        np.std(y_pred) > 0
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


    r2 = r2_score(
        y_true,
        y_pred
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


    return {
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "r2": float(r2),
        "rmse": float(rmse),
    }


# =========================================================
# Models
# =========================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.head = nn.Linear(
            hidden_dim,
            output_dim
        )

        self.dropout = dropout


    def forward(
        self,
        x,
        edge_index=None,
    ):

        x = F.relu(
            self.fc1(x)
        )

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = F.relu(
            self.fc2(x)
        )

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        return self.head(x)


class GraphSAGE(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout,
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.head = nn.Linear(
            hidden_dim,
            output_dim
        )

        self.dropout = dropout


    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        return self.head(x)


def make_model(
    model_name
):

    if model_name == "MLP":

        return MLP(
            PCA_DIM,
            HIDDEN_DIM,
            len(TARGET_GENES),
            DROPOUT,
        )


    if model_name == "GraphSAGE":

        return GraphSAGE(
            PCA_DIM,
            HIDDEN_DIM,
            len(TARGET_GENES),
            DROPOUT,
        )


    raise ValueError(
        model_name
    )


# =========================================================
# Load metadata
# =========================================================

print(
    "\nLoading shared metadata..."
)

manifest = pd.read_csv(
    MANIFEST_PATH
)

split_df = pd.read_csv(
    SPLIT_PATH
)

adata = sc.read_h5ad(
    ADATA_PATH
)


barcodes = (
    manifest[
        "barcode"
    ]
    .astype(str)
    .tolist()
)


adata = adata[
    barcodes
].copy()


split_df[
    "barcode"
] = (
    split_df[
        "barcode"
    ].astype(str)
)


split_df = (
    split_df
    .set_index("barcode")
    .loc[barcodes]
    .reset_index()
)


print(
    "Spots:",
    len(barcodes)
)


# =========================================================
# Expression targets
# =========================================================

Y = np.zeros(
    (
        len(barcodes),
        len(TARGET_GENES)
    ),
    dtype=np.float32,
)


for j, gene in enumerate(
    TARGET_GENES
):

    values = adata[
        :,
        gene
    ].X

    if sparse.issparse(
        values
    ):
        values = values.toarray()


    Y[:, j] = (
        np.asarray(values)
        .ravel()
        .astype(np.float32)
    )


# =========================================================
# Spatial graph
# =========================================================

A = (
    adata.obsp[
        "spatial_connectivities"
    ]
    .tocsr()
    .astype(np.float32)
    .copy()
)

A.setdiag(0)
A.eliminate_zeros()


def get_subgraph(
    adjacency,
    indices,
):

    sub = (
        adjacency[
            indices
        ][:, indices]
        .tocoo()
    )


    edge_index = np.vstack(
        [
            sub.row,
            sub.col,
        ]
    ).astype(np.int64)


    return torch.from_numpy(
        edge_index
    )


# =========================================================
# Fixed strict spatial split
# =========================================================

train_pool_idx = np.where(
    split_df[
        "spatial_split"
    ].values
    ==
    "train"
)[0]


test_idx = np.where(
    split_df[
        "spatial_split"
    ].values
    ==
    "test"
)[0]


print(
    "Train pool:",
    len(train_pool_idx)
)

print(
    "Test:",
    len(test_idx)
)


# =========================================================
# Block-level validation split
# =========================================================

train_groups = (
    split_df.iloc[
        train_pool_idx
    ][
        "spatial_block"
    ].values
)


gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,
    random_state=42,
)


fit_rel, val_rel = next(
    gss.split(
        train_pool_idx,
        groups=train_groups,
    )
)


fit_idx = train_pool_idx[
    fit_rel
]


val_idx = train_pool_idx[
    val_rel
]


edge_fit = get_subgraph(
    A,
    fit_idx
)

edge_val = get_subgraph(
    A,
    val_idx
)

edge_train = get_subgraph(
    A,
    train_pool_idx
)

edge_test = get_subgraph(
    A,
    test_idx
)


print(
    "Fit:",
    len(fit_idx)
)

print(
    "Validation:",
    len(val_idx)
)


# =========================================================
# Neural model selection
# =========================================================

def select_epoch(
    model_name,
    seed,
    X_select_all,
):

    set_seed(seed)


    scaler = StandardScaler()


    X_fit = scaler.fit_transform(
        X_select_all[
            fit_idx
        ]
    ).astype(np.float32)


    X_val = scaler.transform(
        X_select_all[
            val_idx
        ]
    ).astype(np.float32)


    y_fit_raw = Y[
        fit_idx
    ]


    y_val_raw = Y[
        val_idx
    ]


    y_mean = (
        y_fit_raw.mean(
            axis=0
        )
    )


    y_std = (
        y_fit_raw.std(
            axis=0
        )
    )


    y_std[
        y_std < 1e-6
    ] = 1.0


    y_fit = (
        (
            y_fit_raw
            - y_mean
        )
        / y_std
    ).astype(np.float32)


    y_val = (
        (
            y_val_raw
            - y_mean
        )
        / y_std
    ).astype(np.float32)


    x_fit_t = torch.from_numpy(
        X_fit
    ).to(device)


    x_val_t = torch.from_numpy(
        X_val
    ).to(device)


    y_fit_t = torch.from_numpy(
        y_fit
    ).to(device)


    y_val_t = torch.from_numpy(
        y_val
    ).to(device)


    ef = edge_fit.to(
        device
    )

    ev = edge_val.to(
        device
    )


    model = make_model(
        model_name
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    best_val = np.inf
    best_epoch = 1
    bad_epochs = 0


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()


        if model_name == "GraphSAGE":

            pred = model(
                x_fit_t,
                ef
            )

        else:

            pred = model(
                x_fit_t
            )


        loss = F.mse_loss(
            pred,
            y_fit_t
        )


        loss.backward()

        optimizer.step()


        model.eval()


        with torch.no_grad():

            if model_name == "GraphSAGE":

                val_pred = model(
                    x_val_t,
                    ev
                )

            else:

                val_pred = model(
                    x_val_t
                )


            val_loss = F.mse_loss(
                val_pred,
                y_val_t
            ).item()


        if (
            val_loss
            <
            best_val - 1e-6
        ):

            best_val = val_loss

            best_epoch = epoch

            bad_epochs = 0

        else:

            bad_epochs += 1


        if (
            bad_epochs
            >= PATIENCE
        ):
            break


    return (
        best_epoch,
        best_val,
    )


# =========================================================
# Final training
# =========================================================

def train_final(
    model_name,
    seed,
    best_epoch,
    X_final_all,
):

    set_seed(seed)


    scaler = StandardScaler()


    X_train = scaler.fit_transform(
        X_final_all[
            train_pool_idx
        ]
    ).astype(np.float32)


    X_test = scaler.transform(
        X_final_all[
            test_idx
        ]
    ).astype(np.float32)


    y_train_raw = Y[
        train_pool_idx
    ]


    y_mean = y_train_raw.mean(
        axis=0
    )


    y_std = y_train_raw.std(
        axis=0
    )


    y_std[
        y_std < 1e-6
    ] = 1.0


    y_train = (
        (
            y_train_raw
            - y_mean
        )
        / y_std
    ).astype(np.float32)


    x_train_t = torch.from_numpy(
        X_train
    ).to(device)


    x_test_t = torch.from_numpy(
        X_test
    ).to(device)


    y_train_t = torch.from_numpy(
        y_train
    ).to(device)


    etrain = edge_train.to(
        device
    )

    etest = edge_test.to(
        device
    )


    model = make_model(
        model_name
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    for _ in range(
        best_epoch
    ):

        model.train()

        optimizer.zero_grad()


        if model_name == "GraphSAGE":

            pred = model(
                x_train_t,
                etrain
            )

        else:

            pred = model(
                x_train_t
            )


        loss = F.mse_loss(
            pred,
            y_train_t
        )


        loss.backward()

        optimizer.step()


    model.eval()


    with torch.no_grad():

        if model_name == "GraphSAGE":

            pred_scaled = model(
                x_test_t,
                etest
            )

        else:

            pred_scaled = model(
                x_test_t
            )


    pred_scaled = (
        pred_scaled
        .cpu()
        .numpy()
    )


    pred = (
        pred_scaled
        * y_std
        + y_mean
    )


    return pred


# =========================================================
# Run resolution benchmark
# =========================================================

all_rows = []

pca_rows = []


for (
    resolution_name,
    embedding_path
) in RESOLUTIONS.items():

    print(
        "\n======================================"
    )

    print(
        "Resolution:",
        resolution_name
    )

    print(
        "======================================"
    )


    X_raw = np.load(
        embedding_path
    ).astype(np.float32)


    print(
        "Raw embedding:",
        X_raw.shape
    )


    assert (
        X_raw.shape
        ==
        (
            len(manifest),
            384
        )
    )


    assert (
        np.isnan(
            X_raw
        ).sum()
        ==
        0
    )


    # =====================================================
    # PCA for model selection:
    # fit nodes ONLY
    # =====================================================

    pca_select = PCA(
        n_components=PCA_DIM,
        svd_solver="randomized",
        random_state=42,
    )


    pca_select.fit(
        X_raw[
            fit_idx
        ]
    )


    X_select_all = (
        pca_select.transform(
            X_raw
        )
        .astype(np.float32)
    )


    # =====================================================
    # Final PCA:
    # entire training region ONLY
    # =====================================================

    pca_final = PCA(
        n_components=PCA_DIM,
        svd_solver="randomized",
        random_state=42,
    )


    pca_final.fit(
        X_raw[
            train_pool_idx
        ]
    )


    X_final_all = (
        pca_final.transform(
            X_raw
        )
        .astype(np.float32)
    )


    explained = float(
        pca_final
        .explained_variance_ratio_
        .sum()
    )


    print(
        "Train-only PCA variance:",
        explained
    )


    pca_rows.append(
        {
            "resolution":
                resolution_name,

            "raw_dimension":
                X_raw.shape[1],

            "pca_dimension":
                PCA_DIM,

            "train_only_pca_variance":
                explained,
        }
    )


    # =====================================================
    # Ridge
    # =====================================================

    ridge = Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),

            (
                "ridge",
                Ridge(
                    alpha=RIDGE_ALPHA
                )
            ),
        ]
    )


    ridge.fit(
        X_final_all[
            train_pool_idx
        ],
        Y[
            train_pool_idx
        ],
    )


    ridge_pred = ridge.predict(
        X_final_all[
            test_idx
        ]
    )


    for j, gene in enumerate(
        TARGET_GENES
    ):

        metric = evaluate_gene(
            Y[
                test_idx,
                j
            ],
            ridge_pred[
                :,
                j
            ],
        )


        all_rows.append(
            {
                "resolution":
                    resolution_name,

                "model":
                    "Ridge",

                "gene":
                    gene,

                "seed":
                    np.nan,

                **metric,
            }
        )


    # =====================================================
    # MLP + GraphSAGE
    # =====================================================

    for model_name in [
        "MLP",
        "GraphSAGE",
    ]:


        for seed in SEEDS:


            best_epoch, val_loss = (
                select_epoch(
                    model_name,
                    seed,
                    X_select_all,
                )
            )


            print(
                resolution_name,
                model_name,
                f"seed={seed}",
                f"epoch={best_epoch}",
                f"val={val_loss:.4f}",
            )


            pred = train_final(
                model_name,
                seed,
                best_epoch,
                X_final_all,
            )


            for j, gene in enumerate(
                TARGET_GENES
            ):

                metric = evaluate_gene(
                    Y[
                        test_idx,
                        j
                    ],
                    pred[
                        :,
                        j
                    ],
                )


                all_rows.append(
                    {
                        "resolution":
                            resolution_name,

                        "model":
                            model_name,

                        "gene":
                            gene,

                        "seed":
                            seed,

                        **metric,
                    }
                )


# =========================================================
# Save raw metrics
# =========================================================

results = pd.DataFrame(
    all_rows
)


results.to_csv(
    OUTDIR
    / "resolution_strict_raw_metrics.csv",
    index=False,
)


pd.DataFrame(
    pca_rows
).to_csv(
    OUTDIR
    / "resolution_train_only_pca_summary.csv",
    index=False,
)


# =========================================================
# Summary
# =========================================================

summary_rows = []


for (
    resolution,
    model,
    gene
), group in results.groupby(
    [
        "resolution",
        "model",
        "gene",
    ]
):


    summary_rows.append(
        {
            "resolution":
                resolution,

            "model":
                model,

            "gene":
                gene,

            "pearson_mean":
                group[
                    "pearson_r"
                ].mean(),

            "pearson_std":
                (
                    group[
                        "pearson_r"
                    ].std()
                    if len(group) > 1
                    else 0.0
                ),

            "spearman_mean":
                group[
                    "spearman_rho"
                ].mean(),

            "spearman_std":
                (
                    group[
                        "spearman_rho"
                    ].std()
                    if len(group) > 1
                    else 0.0
                ),

            "r2_mean":
                group[
                    "r2"
                ].mean(),

            "r2_std":
                (
                    group[
                        "r2"
                    ].std()
                    if len(group) > 1
                    else 0.0
                ),

            "rmse_mean":
                group[
                    "rmse"
                ].mean(),

            "rmse_std":
                (
                    group[
                        "rmse"
                    ].std()
                    if len(group) > 1
                    else 0.0
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUTDIR
    / "resolution_strict_benchmark.csv",
    index=False,
)


# =========================================================
# Direct FullRes - LowRes comparison
# =========================================================

comparison_rows = []


for model_name in [
    "Ridge",
    "MLP",
    "GraphSAGE",
]:

    for gene in TARGET_GENES:


        sub = summary[
            (
                summary[
                    "model"
                ]
                ==
                model_name
            )
            &
            (
                summary[
                    "gene"
                ]
                ==
                gene
            )
        ]


        low = sub[
            sub[
                "resolution"
            ]
            ==
            "LowRes_38pxUpscaled"
        ].iloc[0]


        full = sub[
            sub[
                "resolution"
            ]
            ==
            "FullRes_224px"
        ].iloc[0]


        comparison_rows.append(
            {
                "model":
                    model_name,

                "gene":
                    gene,

                "lowres_pearson":
                    low[
                        "pearson_mean"
                    ],

                "fullres_pearson":
                    full[
                        "pearson_mean"
                    ],

                "delta_pearson_full_minus_low":
                    (
                        full[
                            "pearson_mean"
                        ]
                        -
                        low[
                            "pearson_mean"
                        ]
                    ),

                "lowres_r2":
                    low[
                        "r2_mean"
                    ],

                "fullres_r2":
                    full[
                        "r2_mean"
                    ],

                "delta_r2_full_minus_low":
                    (
                        full[
                            "r2_mean"
                        ]
                        -
                        low[
                            "r2_mean"
                        ]
                    ),

                "lowres_rmse":
                    low[
                        "rmse_mean"
                    ],

                "fullres_rmse":
                    full[
                        "rmse_mean"
                    ],

                "delta_rmse_full_minus_low":
                    (
                        full[
                            "rmse_mean"
                        ]
                        -
                        low[
                            "rmse_mean"
                        ]
                    ),
            }
        )


comparison = pd.DataFrame(
    comparison_rows
)


comparison.to_csv(
    OUTDIR
    / "fullres_vs_lowres_strict_comparison.csv",
    index=False,
)


print(
    "\n=== Full-resolution vs Low-resolution ==="
)


print(
    comparison.to_string(
        index=False
    )
)


# =========================================================
# Paired seed stability
# =========================================================

neural = results[
    results[
        "model"
    ].isin(
        [
            "MLP",
            "GraphSAGE",
        ]
    )
].copy()


pivot = neural.pivot_table(
    index=[
        "model",
        "gene",
        "seed",
    ],

    columns="resolution",

    values=[
        "pearson_r",
        "r2",
    ],
)


stability_rows = []


for (
    model,
    gene
), group in pivot.groupby(
    level=[
        0,
        1,
    ]
):


    d_r = (
        group[
            (
                "pearson_r",
                "FullRes_224px"
            )
        ]
        -
        group[
            (
                "pearson_r",
                "LowRes_38pxUpscaled"
            )
        ]
    )


    d_r2 = (
        group[
            (
                "r2",
                "FullRes_224px"
            )
        ]
        -
        group[
            (
                "r2",
                "LowRes_38pxUpscaled"
            )
        ]
    )


    stability_rows.append(
        {
            "model":
                model,

            "gene":
                gene,

            "delta_pearson_mean":
                d_r.mean(),

            "delta_pearson_std":
                d_r.std(),

            "fullres_pearson_wins":
                int(
                    (
                        d_r > 0
                    ).sum()
                ),

            "n_seeds":
                len(d_r),

            "delta_r2_mean":
                d_r2.mean(),

            "delta_r2_std":
                d_r2.std(),

            "fullres_r2_wins":
                int(
                    (
                        d_r2 > 0
                    ).sum()
                ),
        }
    )


stability = pd.DataFrame(
    stability_rows
)


stability.to_csv(
    OUTDIR
    / "resolution_paired_seed_stability.csv",
    index=False,
)


# =========================================================
# GraphSAGE Pearson plot
# =========================================================

graph_df = summary[
    summary[
        "model"
    ]
    ==
    "GraphSAGE"
].copy()


x_pos = np.arange(
    len(TARGET_GENES)
)

width = 0.35


fig, ax = plt.subplots(
    figsize=(10, 6)
)


for offset, resolution in [
    (
        -width / 2,
        "LowRes_38pxUpscaled"
    ),
    (
        width / 2,
        "FullRes_224px"
    ),
]:

    values = []
    errors = []


    for gene in TARGET_GENES:

        row = graph_df[
            (
                graph_df[
                    "resolution"
                ]
                ==
                resolution
            )
            &
            (
                graph_df[
                    "gene"
                ]
                ==
                gene
            )
        ].iloc[0]


        values.append(
            row[
                "pearson_mean"
            ]
        )


        errors.append(
            row[
                "pearson_std"
            ]
        )


    ax.bar(
        x_pos + offset,
        values,
        width,
        yerr=errors,
        label=resolution,
    )


ax.set_xticks(
    x_pos
)

ax.set_xticklabels(
    TARGET_GENES
)

ax.set_ylabel(
    "Pearson r"
)

ax.set_title(
    "DINOv2 + GraphSAGE:\n"
    "low-resolution vs full-resolution H&E"
)

ax.legend()


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "34_lowres_vs_fullres_graphsage.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# Delta Pearson plot
# =========================================================

graph_comparison = comparison[
    comparison[
        "model"
    ]
    ==
    "GraphSAGE"
].copy()


fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(
    graph_comparison[
        "gene"
    ],
    graph_comparison[
        "delta_pearson_full_minus_low"
    ],
)


ax.axhline(
    0,
    linewidth=1,
)


ax.set_ylabel(
    "Δ Pearson r\n"
    "(Full-res - Low-res)"
)


ax.set_title(
    "Effect of true histological resolution"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "35_fullres_resolution_delta_pearson.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# Final report
# =========================================================

print(
    "\n=== Paired seed stability ==="
)

print(
    stability.to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    "outputs/resolution_strict_raw_metrics.csv"
)

print(
    "outputs/resolution_train_only_pca_summary.csv"
)

print(
    "outputs/resolution_strict_benchmark.csv"
)

print(
    "outputs/fullres_vs_lowres_strict_comparison.csv"
)

print(
    "outputs/resolution_paired_seed_stability.csv"
)

print(
    "figures/34_lowres_vs_fullres_graphsage.png"
)

print(
    "figures/35_fullres_resolution_delta_pearson.png"
)


print(
    "\nStrict resolution ablation complete."
)
