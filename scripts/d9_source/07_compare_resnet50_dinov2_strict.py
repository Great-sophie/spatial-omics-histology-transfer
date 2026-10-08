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

ENCODERS = {
    "ResNet50": Path(
        "outputs/resnet50_image_embeddings.npy"
    ),
    "DINOv2_ViTS14": Path(
        "outputs/dinov2_vits14_image_embeddings.npy"
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
            y_pred,
        ).statistic

        spearman = spearmanr(
            y_true,
            y_pred,
        ).statistic

    else:

        pearson = np.nan
        spearman = np.nan


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


    return {
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "r2": float(r2),
        "rmse": float(rmse),
    }


# =========================================================
# Neural models
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
            hidden_dim,
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.head = nn.Linear(
            hidden_dim,
            output_dim,
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
            hidden_dim,
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim,
        )

        self.head = nn.Linear(
            hidden_dim,
            output_dim,
        )

        self.dropout = dropout


    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        return self.head(x)


def make_model(model_name):

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

    raise ValueError(model_name)


# =========================================================
# Load shared metadata
# =========================================================

print("\nLoading metadata...")

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
    manifest["barcode"]
    .astype(str)
    .tolist()
)


adata = adata[
    barcodes
].copy()


split_df["barcode"] = (
    split_df["barcode"]
    .astype(str)
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
        len(TARGET_GENES),
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

    if sparse.issparse(values):

        values = values.toarray()

    values = (
        np.asarray(values)
        .ravel()
        .astype(np.float32)
    )

    Y[:, j] = values


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
# Fixed train/test split
# =========================================================

train_pool_idx = np.where(
    split_df[
        "spatial_split"
    ].values
    == "train"
)[0]


test_idx = np.where(
    split_df[
        "spatial_split"
    ].values
    == "test"
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
# Train/validation block split
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
    fit_idx,
)

edge_val = get_subgraph(
    A,
    val_idx,
)

edge_train = get_subgraph(
    A,
    train_pool_idx,
)

edge_test = get_subgraph(
    A,
    test_idx,
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
# Model selection
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


    y_mean = y_fit_raw.mean(
        axis=0
    )


    y_std = y_fit_raw.std(
        axis=0
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


    ef = edge_fit.to(device)

    ev = edge_val.to(device)


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
                ef,
            )

        else:

            pred = model(
                x_fit_t
            )


        loss = F.mse_loss(
            pred,
            y_fit_t,
        )


        loss.backward()

        optimizer.step()


        model.eval()


        with torch.no_grad():

            if model_name == "GraphSAGE":

                val_pred = model(
                    x_val_t,
                    ev,
                )

            else:

                val_pred = model(
                    x_val_t
                )


            val_loss = F.mse_loss(
                val_pred,
                y_val_t,
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


        if bad_epochs >= PATIENCE:
            break


    return (
        best_epoch,
        best_val,
    )


# =========================================================
# Final neural training
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


    for epoch in range(
        best_epoch
    ):

        model.train()

        optimizer.zero_grad()


        if model_name == "GraphSAGE":

            pred = model(
                x_train_t,
                etrain,
            )

        else:

            pred = model(
                x_train_t
            )


        loss = F.mse_loss(
            pred,
            y_train_t,
        )


        loss.backward()

        optimizer.step()


    model.eval()


    with torch.no_grad():

        if model_name == "GraphSAGE":

            pred_scaled = model(
                x_test_t,
                etest,
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
# Benchmark encoders
# =========================================================

all_rows = []

pca_rows = []


for encoder_name, embedding_path in ENCODERS.items():

    print(
        "\n======================================"
    )

    print(
        "Encoder:",
        encoder_name
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
        X_raw.shape[0]
        ==
        len(manifest)
    )


    # -----------------------------------------------------
    # Selection PCA: fit nodes only
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Final PCA: train pool only
    # -----------------------------------------------------

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


    pca_variance = float(
        pca_final
        .explained_variance_ratio_
        .sum()
    )


    print(
        "Train-only PCA variance:",
        pca_variance
    )


    pca_rows.append(
        {
            "encoder":
                encoder_name,

            "raw_dimension":
                X_raw.shape[1],

            "pca_dimension":
                PCA_DIM,

            "train_only_pca_variance":
                pca_variance,
        }
    )


    # -----------------------------------------------------
    # Ridge baseline
    # -----------------------------------------------------

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

        m = evaluate_gene(
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
                "encoder":
                    encoder_name,

                "model":
                    "Ridge",

                "gene":
                    gene,

                "seed":
                    np.nan,

                "best_epoch":
                    np.nan,

                **m,
            }
        )


    # -----------------------------------------------------
    # MLP / GraphSAGE
    # -----------------------------------------------------

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
                encoder_name,
                model_name,
                "seed=",
                seed,
                "epoch=",
                best_epoch,
                "val=",
                round(
                    val_loss,
                    4
                ),
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

                m = evaluate_gene(
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
                        "encoder":
                            encoder_name,

                        "model":
                            model_name,

                        "gene":
                            gene,

                        "seed":
                            seed,

                        "best_epoch":
                            best_epoch,

                        **m,
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
    / "encoder_strict_raw_metrics.csv",
    index=False,
)


pd.DataFrame(
    pca_rows
).to_csv(
    OUTDIR
    / "encoder_train_only_pca_summary.csv",
    index=False,
)


# =========================================================
# Summarize
# =========================================================

summary_rows = []


for (
    encoder,
    model,
    gene
), group in results.groupby(
    [
        "encoder",
        "model",
        "gene",
    ]
):

    summary_rows.append(
        {
            "encoder":
                encoder,

            "model":
                model,

            "gene":
                gene,

            "pearson_mean":
                group[
                    "pearson_r"
                ].mean(),

            "pearson_std":
                group[
                    "pearson_r"
                ].std(
                    ddof=1
                )
                if len(group) > 1
                else 0.0,

            "spearman_mean":
                group[
                    "spearman_rho"
                ].mean(),

            "spearman_std":
                group[
                    "spearman_rho"
                ].std(
                    ddof=1
                )
                if len(group) > 1
                else 0.0,

            "r2_mean":
                group[
                    "r2"
                ].mean(),

            "r2_std":
                group[
                    "r2"
                ].std(
                    ddof=1
                )
                if len(group) > 1
                else 0.0,

            "rmse_mean":
                group[
                    "rmse"
                ].mean(),

            "rmse_std":
                group[
                    "rmse"
                ].std(
                    ddof=1
                )
                if len(group) > 1
                else 0.0,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUTDIR
    / "encoder_strict_benchmark.csv",
    index=False,
)


print(
    "\n=== Encoder benchmark ==="
)


print(
    summary.sort_values(
        [
            "gene",
            "model",
            "encoder",
        ]
    ).to_string(
        index=False
    )
)


# =========================================================
# Direct DINO - ResNet comparison
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


        resnet = sub[
            sub[
                "encoder"
            ]
            ==
            "ResNet50"
        ].iloc[0]


        dino = sub[
            sub[
                "encoder"
            ]
            ==
            "DINOv2_ViTS14"
        ].iloc[0]


        comparison_rows.append(
            {
                "model":
                    model_name,

                "gene":
                    gene,

                "resnet_pearson":
                    resnet[
                        "pearson_mean"
                    ],

                "dinov2_pearson":
                    dino[
                        "pearson_mean"
                    ],

                "delta_pearson_dino_minus_resnet":
                    (
                        dino[
                            "pearson_mean"
                        ]
                        -
                        resnet[
                            "pearson_mean"
                        ]
                    ),

                "resnet_r2":
                    resnet[
                        "r2_mean"
                    ],

                "dinov2_r2":
                    dino[
                        "r2_mean"
                    ],

                "delta_r2_dino_minus_resnet":
                    (
                        dino[
                            "r2_mean"
                        ]
                        -
                        resnet[
                            "r2_mean"
                        ]
                    ),
            }
        )


comparison = pd.DataFrame(
    comparison_rows
)


comparison.to_csv(
    OUTDIR
    / "dinov2_vs_resnet50_strict_comparison.csv",
    index=False,
)


print(
    "\n=== DINOv2 - ResNet50 ==="
)


print(
    comparison.to_string(
        index=False
    )
)


# =========================================================
# Plot GraphSAGE comparison
# =========================================================

graph_df = summary[
    summary[
        "model"
    ]
    ==
    "GraphSAGE"
].copy()


x = np.arange(
    len(TARGET_GENES)
)

width = 0.35


fig, ax = plt.subplots(
    figsize=(10, 6)
)


for offset, encoder in [
    (-width / 2, "ResNet50"),
    ( width / 2, "DINOv2_ViTS14"),
]:

    values = []

    errors = []


    for gene in TARGET_GENES:

        row = graph_df[
            (
                graph_df[
                    "encoder"
                ]
                ==
                encoder
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
        x + offset,
        values,
        width,
        yerr=errors,
        label=encoder,
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
    "Strict spatial-block GraphSAGE:\n"
    "ResNet50 vs DINOv2"
)

ax.legend()


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "30_resnet50_vs_dinov2_graphsage.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# Delta plot
# =========================================================

graph_comp = comparison[
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
    graph_comp[
        "gene"
    ],
    graph_comp[
        "delta_pearson_dino_minus_resnet"
    ],
)


ax.axhline(
    0,
    linewidth=1,
)


ax.set_ylabel(
    "Δ Pearson r\n(DINOv2 - ResNet50)"
)

ax.set_title(
    "Encoder effect under strict GraphSAGE evaluation"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "31_dinov2_minus_resnet50_delta.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


print("\nSaved:")

print(
    "outputs/encoder_strict_raw_metrics.csv"
)

print(
    "outputs/encoder_train_only_pca_summary.csv"
)

print(
    "outputs/encoder_strict_benchmark.csv"
)

print(
    "outputs/dinov2_vs_resnet50_strict_comparison.csv"
)

print(
    "figures/30_resnet50_vs_dinov2_graphsage.png"
)

print(
    "figures/31_dinov2_minus_resnet50_delta.png"
)


print(
    "\nStrict encoder benchmark complete."
)
