import copy
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

RAW_EMBEDDING_PATH = Path(
    "outputs/resnet50_image_embeddings.npy"
)

MANIFEST_PATH = Path(
    "outputs/spot_image_embedding_manifest.csv"
)

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
# 1. Load data
# =========================================================

print("\nLoading raw morphology embeddings...")

X_raw = np.load(
    RAW_EMBEDDING_PATH
).astype(np.float32)


manifest = pd.read_csv(
    MANIFEST_PATH
)


split_df = pd.read_csv(
    SPLIT_PATH
)


print(
    "Raw embeddings:",
    X_raw.shape
)

print(
    "Manifest:",
    manifest.shape
)

print(
    "Split manifest:",
    split_df.shape
)


assert (
    X_raw.shape[0]
    == len(manifest)
)

assert (
    np.isnan(X_raw).sum()
    == 0
)


# =========================================================
# 2. Load spatial AnnData
# =========================================================

print("\nLoading spatial AnnData...")

adata = sc.read_h5ad(
    ADATA_PATH
)


print(adata)


if (
    "spatial_connectivities"
    not in adata.obsp
):

    raise KeyError(
        "spatial_connectivities "
        "not found in AnnData."
    )


# =========================================================
# 3. Align by barcode
# =========================================================

barcodes = (
    manifest[
        "barcode"
    ]
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
    .set_index(
        "barcode"
    )
    .loc[
        barcodes
    ]
    .reset_index()
)


assert (
    list(
        adata.obs_names
    )
    ==
    barcodes
)


assert (
    list(
        split_df[
            "barcode"
        ]
    )
    ==
    barcodes
)


print(
    "\nBarcode alignment successful:",
    len(barcodes),
    "spots"
)


# =========================================================
# 4. Load spatial graph
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


print("\nFull spatial graph:")

print(
    "shape:",
    A.shape
)

print(
    "directed edge entries:",
    A.nnz
)


# =========================================================
# 5. Expression targets
#
# Use normalized/log-transformed adata.X.
# =========================================================

Y = np.zeros(
    (
        adata.n_obs,
        len(TARGET_GENES)
    ),
    dtype=np.float32,
)


print(
    "\nExpression targets:"
)


for j, gene in enumerate(
    TARGET_GENES
):

    if gene not in adata.var_names:

        raise KeyError(
            f"{gene} not found."
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
        np.asarray(values)
        .ravel()
        .astype(np.float32)
    )


    Y[
        :,
        j
    ] = values


    print(
        f"{gene}: "
        f"mean={values.mean():.4f}, "
        f"std={values.std():.4f}, "
        f"min={values.min():.4f}, "
        f"max={values.max():.4f}"
    )


# =========================================================
# 6. Fixed spatial train/test split
#
# Reuse EXACT split from D9.
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
    "\nFixed spatial split:"
)

print(
    "Train pool:",
    len(train_pool_idx)
)

print(
    "Test:",
    len(test_idx)
)


print(
    "Test blocks:",
    sorted(
        split_df.iloc[
            test_idx
        ][
            "spatial_block"
        ].unique()
    )
)


# =========================================================
# 7. Block-level validation split
#
# Validation comes only from training blocks.
# Test blocks remain completely untouched.
# =========================================================

train_groups = (
    split_df.iloc[
        train_pool_idx
    ][
        "spatial_block"
    ]
    .values
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


fit_idx = (
    train_pool_idx[
        fit_rel
    ]
)


val_idx = (
    train_pool_idx[
        val_rel
    ]
)


print(
    "\nModel-selection split:"
)

print(
    "Fit:",
    len(fit_idx)
)

print(
    "Validation:",
    len(val_idx)
)


print(
    "Validation blocks:",
    sorted(
        split_df.iloc[
            val_idx
        ][
            "spatial_block"
        ].unique()
    )
)


# =========================================================
# 8. Graph helper
#
# Subsetting removes ALL cross-split edges.
# =========================================================

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


    edge_index = torch.from_numpy(
        edge_index
    )


    return (
        sub.tocsr(),
        edge_index,
    )


A_fit, edge_fit = get_subgraph(
    A,
    fit_idx
)


A_val, edge_val = get_subgraph(
    A,
    val_idx
)


A_train, edge_train = get_subgraph(
    A,
    train_pool_idx
)


A_test, edge_test = get_subgraph(
    A,
    test_idx
)


print(
    "\nStrict graphs:"
)

print(
    "Fit edges:",
    A_fit.nnz
)

print(
    "Validation edges:",
    A_val.nnz
)

print(
    "Full training edges:",
    A_train.nnz
)

print(
    "Test-only edges:",
    A_test.nnz
)


# =========================================================
# 9. PCA for model selection
#
# Fit ONLY on fit nodes.
# Validation does not influence PCA.
# =========================================================

print(
    "\nFitting model-selection PCA..."
)


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


print(
    "Selection PCA variance:",
    float(
        pca_select
        .explained_variance_ratio_
        .sum()
    )
)


# =========================================================
# 10. Final PCA
#
# After model selection, fit PCA on ALL training nodes.
# Test nodes remain completely excluded.
# =========================================================

print(
    "\nFitting final train-only PCA..."
)


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


print(
    "Final train-only PCA variance:",
    float(
        pca_final
        .explained_variance_ratio_
        .sum()
    )
)


np.save(
    OUTDIR
    / "strict_train_only_pca_features.npy",
    X_final_all,
)


np.save(
    OUTDIR
    / "strict_train_only_pca_components.npy",
    pca_final.components_,
)


# =========================================================
# 11. Strict 1-hop morphology aggregation
# =========================================================

def neighbor_mean(
    adjacency,
    features,
):

    degree = np.asarray(
        adjacency.sum(
            axis=1
        )
    ).ravel()


    safe_degree = (
        degree.copy()
    )


    safe_degree[
        safe_degree == 0
    ] = 1.0


    W = (
        sparse.diags(
            1.0
            / safe_degree
        )
        @ adjacency
    )


    out = (
        W
        @ features
    )


    return (
        np.asarray(
            out
        )
        .astype(np.float32)
    )


X_train_center = (
    X_final_all[
        train_pool_idx
    ]
)


X_test_center = (
    X_final_all[
        test_idx
    ]
)


X_train_neighbor = neighbor_mean(
    A_train,
    X_train_center,
)


X_test_neighbor = neighbor_mean(
    A_test,
    X_test_center,
)


X_train_context = np.concatenate(
    [
        X_train_center,
        X_train_neighbor,
    ],
    axis=1,
)


X_test_context = np.concatenate(
    [
        X_test_center,
        X_test_neighbor,
    ],
    axis=1,
)


print(
    "\nStrict context features:"
)

print(
    "Train:",
    X_train_context.shape
)

print(
    "Test:",
    X_test_context.shape
)


# =========================================================
# 12. Metrics
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
        "pearson_r":
            float(pearson),

        "spearman_rho":
            float(spearman),

        "r2":
            float(r2),

        "rmse":
            float(rmse),
    }


def evaluate_matrix(
    y_true,
    y_pred,
    model_name,
    seed=None,
):

    rows = []


    for j, gene in enumerate(
        TARGET_GENES
    ):

        result = evaluate_gene(
            y_true[
                :,
                j
            ],
            y_pred[
                :,
                j
            ],
        )


        rows.append(
            {
                "gene":
                    gene,

                "model":
                    model_name,

                "seed":
                    seed,

                **result,
            }
        )


    return rows


# =========================================================
# 13. Strict Ridge morphology baseline
# =========================================================

print(
    "\n======================================"
)

print(
    "Strict Ridge morphology baseline"
)

print(
    "======================================"
)


ridge_morph = Pipeline(
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


ridge_morph.fit(
    X_train_center,
    Y[
        train_pool_idx
    ],
)


pred_ridge_morph = (
    ridge_morph.predict(
        X_test_center
    )
)


baseline_rows = evaluate_matrix(
    Y[
        test_idx
    ],
    pred_ridge_morph,
    "RidgeMorphology",
)


# =========================================================
# 14. Strict 1-hop Ridge baseline
# =========================================================

print(
    "\n======================================"
)

print(
    "Strict Ridge + 1-hop spatial context"
)

print(
    "======================================"
)


ridge_context = Pipeline(
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


ridge_context.fit(
    X_train_context,
    Y[
        train_pool_idx
    ],
)


pred_ridge_context = (
    ridge_context.predict(
        X_test_context
    )
)


baseline_rows += evaluate_matrix(
    Y[
        test_idx
    ],
    pred_ridge_context,
    "Ridge1HopContext",
)


baseline_df = pd.DataFrame(
    baseline_rows
)


print(
    baseline_df
    .to_string(
        index=False
    )
)


# =========================================================
# 15. Neural models
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

        x = self.fc1(x)

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )


        x = self.fc2(x)

        x = F.relu(x)

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
    model_name,
):

    if model_name == "MLP":

        return MLP(
            input_dim=PCA_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=len(
                TARGET_GENES
            ),
            dropout=DROPOUT,
        )


    if model_name == "GraphSAGE":

        return GraphSAGE(
            input_dim=PCA_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=len(
                TARGET_GENES
            ),
            dropout=DROPOUT,
        )


    raise ValueError(
        model_name
    )


# =========================================================
# 16. Model-selection training
#
# Fit graph and validation graph are separate.
# No fit↔validation graph edges.
# =========================================================

def select_best_epoch(
    model_name,
    seed,
):

    set_seed(seed)


    # Feature normalization:
    # fit nodes only.

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


    # Target normalization:
    # fit nodes only.

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


    edge_fit_t = (
        edge_fit
        .to(device)
    )


    edge_val_t = (
        edge_val
        .to(device)
    )


    model = (
        make_model(
            model_name
        )
        .to(device)
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    best_val = np.inf

    best_epoch = 1

    epochs_without_improvement = 0

    history = []


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()


        if model_name == "GraphSAGE":

            pred = model(
                x_fit_t,
                edge_fit_t
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
                    edge_val_t
                )

            else:

                val_pred = model(
                    x_val_t
                )


            val_loss = F.mse_loss(
                val_pred,
                y_val_t
            ).item()


        history.append(
            [
                epoch,
                float(
                    loss.item()
                ),
                float(
                    val_loss
                ),
            ]
        )


        if (
            val_loss
            <
            best_val - 1e-6
        ):

            best_val = val_loss

            best_epoch = epoch

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1


        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            break


    print(
        f"{model_name} seed={seed}"
        f" best_epoch={best_epoch}"
        f" val_loss={best_val:.4f}"
    )


    return (
        best_epoch,
        history,
    )


# =========================================================
# 17. Final training
#
# After choosing epoch count:
#
# - PCA was fit on full training pool only.
# - scaler fit on full training pool only.
# - model trained on full training graph only.
# - test graph remains isolated.
# =========================================================

def train_final_and_predict(
    model_name,
    seed,
    best_epoch,
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


    y_mean = (
        y_train_raw.mean(
            axis=0
        )
    )


    y_std = (
        y_train_raw.std(
            axis=0
        )
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


    edge_train_t = (
        edge_train
        .to(device)
    )


    edge_test_t = (
        edge_test
        .to(device)
    )


    model = (
        make_model(
            model_name
        )
        .to(device)
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    for epoch in range(
        1,
        best_epoch + 1
    ):

        model.train()

        optimizer.zero_grad()


        if model_name == "GraphSAGE":

            pred = model(
                x_train_t,
                edge_train_t
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
                edge_test_t
            )

        else:

            pred_scaled = model(
                x_test_t
            )


    pred_scaled = (
        pred_scaled
        .detach()
        .cpu()
        .numpy()
    )


    pred = (
        pred_scaled
        * y_std
        + y_mean
    )


    return (
        pred.astype(np.float32)
    )


# =========================================================
# 18. Train MLP and GraphSAGE across seeds
# =========================================================

neural_rows = []

prediction_store = {
    "MLP": [],
    "GraphSAGE": [],
}


graphsage_history = None


for model_name in [
    "MLP",
    "GraphSAGE",
]:

    print(
        "\n======================================"
    )

    print(
        model_name
    )

    print(
        "======================================"
    )


    for seed in SEEDS:

        best_epoch, history = (
            select_best_epoch(
                model_name,
                seed,
            )
        )


        if (
            model_name == "GraphSAGE"
            and
            seed == SEEDS[0]
        ):

            graphsage_history = history


        pred = train_final_and_predict(
            model_name,
            seed,
            best_epoch,
        )


        prediction_store[
            model_name
        ].append(
            pred
        )


        rows = evaluate_matrix(
            Y[
                test_idx
            ],
            pred,
            model_name,
            seed=seed,
        )


        for row in rows:

            row[
                "best_epoch"
            ] = best_epoch


        neural_rows.extend(
            rows
        )


neural_df = pd.DataFrame(
    neural_rows
)


neural_df.to_csv(
    OUTDIR
    / "graph_neural_seed_metrics.csv",
    index=False,
)


# =========================================================
# 19. Neural summary across seeds
# =========================================================

summary = (
    neural_df
    .groupby(
        [
            "model",
            "gene",
        ]
    )
    .agg(
        pearson_mean=(
            "pearson_r",
            "mean"
        ),

        pearson_std=(
            "pearson_r",
            "std"
        ),

        spearman_mean=(
            "spearman_rho",
            "mean"
        ),

        spearman_std=(
            "spearman_rho",
            "std"
        ),

        r2_mean=(
            "r2",
            "mean"
        ),

        r2_std=(
            "r2",
            "std"
        ),

        rmse_mean=(
            "rmse",
            "mean"
        ),

        rmse_std=(
            "rmse",
            "std"
        ),

        epoch_mean=(
            "best_epoch",
            "mean"
        ),
    )
    .reset_index()
)


summary.to_csv(
    OUTDIR
    / "graph_neural_summary.csv",
    index=False,
)


print(
    "\n=== Neural model summary ==="
)


print(
    summary.to_string(
        index=False
    )
)


# =========================================================
# 20. Ensemble predictions
#
# Mean prediction across five seeds.
# =========================================================

ensemble_predictions = {}


for model_name in [
    "MLP",
    "GraphSAGE",
]:

    ensemble_predictions[
        model_name
    ] = np.mean(
        np.stack(
            prediction_store[
                model_name
            ],
            axis=0,
        ),
        axis=0,
    )


# =========================================================
# 21. Final benchmark table
# =========================================================

benchmark_rows = []


# Deterministic Ridge baselines

for _, row in baseline_df.iterrows():

    benchmark_rows.append(
        {
            "gene":
                row[
                    "gene"
                ],

            "model":
                row[
                    "model"
                ],

            "pearson_mean":
                row[
                    "pearson_r"
                ],

            "pearson_std":
                0.0,

            "spearman_mean":
                row[
                    "spearman_rho"
                ],

            "spearman_std":
                0.0,

            "r2_mean":
                row[
                    "r2"
                ],

            "r2_std":
                0.0,

            "rmse_mean":
                row[
                    "rmse"
                ],

            "rmse_std":
                0.0,
        }
    )


# Neural models

for _, row in summary.iterrows():

    benchmark_rows.append(
        {
            "gene":
                row[
                    "gene"
                ],

            "model":
                row[
                    "model"
                ],

            "pearson_mean":
                row[
                    "pearson_mean"
                ],

            "pearson_std":
                row[
                    "pearson_std"
                ],

            "spearman_mean":
                row[
                    "spearman_mean"
                ],

            "spearman_std":
                row[
                    "spearman_std"
                ],

            "r2_mean":
                row[
                    "r2_mean"
                ],

            "r2_std":
                row[
                    "r2_std"
                ],

            "rmse_mean":
                row[
                    "rmse_mean"
                ],

            "rmse_std":
                row[
                    "rmse_std"
                ],
        }
    )


benchmark = pd.DataFrame(
    benchmark_rows
)


model_order = {
    "RidgeMorphology": 0,
    "Ridge1HopContext": 1,
    "MLP": 2,
    "GraphSAGE": 3,
}


gene_order = {
    gene: i
    for i, gene
    in enumerate(
        TARGET_GENES
    )
}


benchmark[
    "_model_order"
] = benchmark[
    "model"
].map(
    model_order
)


benchmark[
    "_gene_order"
] = benchmark[
    "gene"
].map(
    gene_order
)


benchmark = (
    benchmark
    .sort_values(
        [
            "_gene_order",
            "_model_order",
        ]
    )
    .drop(
        columns=[
            "_gene_order",
            "_model_order",
        ]
    )
)


benchmark.to_csv(
    OUTDIR
    / "strict_graph_model_benchmark.csv",
    index=False,
)


print(
    "\n=== Strict graph benchmark ==="
)


print(
    benchmark.to_string(
        index=False
    )
)


# =========================================================
# 22. Save test predictions
# =========================================================

prediction_df = pd.DataFrame(
    {
        "barcode":
            manifest.iloc[
                test_idx
            ][
                "barcode"
            ].values,

        "fullres_x":
            manifest.iloc[
                test_idx
            ][
                "fullres_x"
            ].values,

        "fullres_y":
            manifest.iloc[
                test_idx
            ][
                "fullres_y"
            ].values,
    }
)


for j, gene in enumerate(
    TARGET_GENES
):

    prediction_df[
        f"{gene}_true"
    ] = Y[
        test_idx,
        j
    ]


    prediction_df[
        f"{gene}_RidgeMorphology"
    ] = pred_ridge_morph[
        :,
        j
    ]


    prediction_df[
        f"{gene}_Ridge1HopContext"
    ] = pred_ridge_context[
        :,
        j
    ]


    prediction_df[
        f"{gene}_MLP"
    ] = ensemble_predictions[
        "MLP"
    ][
        :,
        j
    ]


    prediction_df[
        f"{gene}_GraphSAGE"
    ] = ensemble_predictions[
        "GraphSAGE"
    ][
        :,
        j
    ]


prediction_df.to_csv(
    OUTDIR
    / "strict_graph_test_predictions.csv",
    index=False,
)


# =========================================================
# 23. Benchmark Pearson plot
# =========================================================

models_plot = [
    "RidgeMorphology",
    "Ridge1HopContext",
    "MLP",
    "GraphSAGE",
]


x = np.arange(
    len(
        TARGET_GENES
    )
)


width = 0.20


fig, ax = plt.subplots(
    figsize=(11, 6)
)


for k, model_name in enumerate(
    models_plot
):

    vals = []

    errs = []


    for gene in TARGET_GENES:

        row = benchmark[
            (
                benchmark[
                    "gene"
                ]
                ==
                gene
            )
            &
            (
                benchmark[
                    "model"
                ]
                ==
                model_name
            )
        ].iloc[0]


        vals.append(
            row[
                "pearson_mean"
            ]
        )


        errs.append(
            row[
                "pearson_std"
            ]
        )


    offset = (
        k
        -
        (
            len(
                models_plot
            )
            - 1
        )
        / 2
    ) * width


    ax.bar(
        x + offset,
        vals,
        width,
        yerr=errs,
        label=model_name,
    )


ax.set_xticks(
    x
)


ax.set_xticklabels(
    TARGET_GENES
)


ax.set_ylabel(
    "Pearson r"
)


ax.set_title(
    "Strict spatial-block molecular prediction"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "23_strict_graph_benchmark_pearson.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 24. Benchmark R² plot
# =========================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)


for k, model_name in enumerate(
    models_plot
):

    vals = []

    errs = []


    for gene in TARGET_GENES:

        row = benchmark[
            (
                benchmark[
                    "gene"
                ]
                ==
                gene
            )
            &
            (
                benchmark[
                    "model"
                ]
                ==
                model_name
            )
        ].iloc[0]


        vals.append(
            row[
                "r2_mean"
            ]
        )


        errs.append(
            row[
                "r2_std"
            ]
        )


    offset = (
        k
        -
        (
            len(
                models_plot
            )
            - 1
        )
        / 2
    ) * width


    ax.bar(
        x + offset,
        vals,
        width,
        yerr=errs,
        label=model_name,
    )


ax.axhline(
    0,
    linewidth=1,
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    TARGET_GENES
)


ax.set_ylabel(
    "R²"
)


ax.set_title(
    "Strict spatial-block molecular prediction"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "24_strict_graph_benchmark_r2.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 25. GraphSAGE learning curve
# =========================================================

if graphsage_history is not None:

    hist = np.asarray(
        graphsage_history
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.plot(
        hist[
            :,
            0
        ],
        hist[
            :,
            1
        ],
        label="Training loss",
    )


    ax.plot(
        hist[
            :,
            0
        ],
        hist[
            :,
            2
        ],
        label="Validation loss",
    )


    ax.set_xlabel(
        "Epoch"
    )


    ax.set_ylabel(
        "Standardized MSE"
    )


    ax.set_title(
        "GraphSAGE learning curve"
    )


    ax.legend()


    plt.tight_layout()


    plt.savefig(
        FIGDIR
        / "25_graphsage_learning_curve.png",
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


# =========================================================
# 26-29. GraphSAGE spatial prediction maps
# =========================================================

graph_pred = ensemble_predictions[
    "GraphSAGE"
]


for j, gene in enumerate(
    TARGET_GENES
):

    x_coord = (
        manifest.iloc[
            test_idx
        ][
            "fullres_x"
        ].values
    )


    y_coord = (
        manifest.iloc[
            test_idx
        ][
            "fullres_y"
        ].values
    )


    true = Y[
        test_idx,
        j
    ]


    pred = graph_pred[
        :,
        j
    ]


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


    error_limit = max(
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
        x_coord,
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
        x_coord,
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
        f"{gene}: GraphSAGE"
    )


    s2 = axes[
        2
    ].scatter(
        x_coord,
        y_coord,
        c=error,
        s=14,
        vmin=-error_limit,
        vmax=error_limit,
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
        f"{gene} | strict inductive GraphSAGE"
    )


    plt.savefig(
        FIGDIR
        / (
            f"{26+j:02d}_"
            f"{gene}_"
            "strict_graphsage_prediction.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )


    plt.close()


# =========================================================
# 30. Save experiment metadata
# =========================================================

experiment_summary = pd.DataFrame(
    {
        "metric": [
            "n_total",
            "n_train_pool",
            "n_test",
            "n_fit",
            "n_validation",
            "raw_embedding_dim",
            "pca_dim",
            "final_pca_variance",
            "hidden_dim",
            "dropout",
            "learning_rate",
            "weight_decay",
            "max_epochs",
            "patience",
            "n_seeds",
            "graph_training_mode",
        ],

        "value": [
            len(manifest),
            len(train_pool_idx),
            len(test_idx),
            len(fit_idx),
            len(val_idx),
            X_raw.shape[1],
            PCA_DIM,
            float(
                pca_final
                .explained_variance_ratio_
                .sum()
            ),
            HIDDEN_DIM,
            DROPOUT,
            LEARNING_RATE,
            WEIGHT_DECAY,
            MAX_EPOCHS,
            PATIENCE,
            len(SEEDS),
            (
                "strict_inductive_"
                "train_test_graph_separation"
            ),
        ],
    }
)


experiment_summary.to_csv(
    OUTDIR
    / "graphsage_experiment_summary.csv",
    index=False,
)


print("\nSaved:")

print(
    "outputs/strict_train_only_pca_features.npy"
)

print(
    "outputs/graph_neural_seed_metrics.csv"
)

print(
    "outputs/graph_neural_summary.csv"
)

print(
    "outputs/strict_graph_model_benchmark.csv"
)

print(
    "outputs/strict_graph_test_predictions.csv"
)

print(
    "outputs/graphsage_experiment_summary.csv"
)

print(
    "figures/23_strict_graph_benchmark_pearson.png"
)

print(
    "figures/24_strict_graph_benchmark_r2.png"
)

print(
    "figures/25_graphsage_learning_curve.png"
)

print(
    "figures/26-29 GraphSAGE spatial maps"
)


print(
    "\nStrict GraphSAGE experiment complete."
)
