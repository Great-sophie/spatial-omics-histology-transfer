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
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)

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

LOWRES_EMBEDDING_PATH = Path(
    "outputs/dinov2_vits14_image_embeddings.npy"
)

FULLRES_EMBEDDING_PATH = Path(
    "outputs/dinov2_vits14_fullres_image_embeddings.npy"
)

LOWRES_MANIFEST_PATH = Path(
    "outputs/spot_image_embedding_manifest.csv"
)

FULLRES_MANIFEST_PATH = Path(
    "outputs/fullres_spot_patch_manifest.csv"
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
# Metrics
# =========================================================

def evaluate_gene(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )


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

        "pearson_r":
            float(pearson),

        "spearman_rho":
            float(spearman),

        "r2":
            float(r2),

        "rmse":
            float(rmse),
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
# 1. Load embeddings
# =========================================================

print(
    "\nLoading DINOv2 embeddings..."
)


X_low = np.load(
    LOWRES_EMBEDDING_PATH
).astype(np.float32)


X_full = np.load(
    FULLRES_EMBEDDING_PATH
).astype(np.float32)


print(
    "Low-res:",
    X_low.shape
)

print(
    "Full-res:",
    X_full.shape
)


assert (
    X_low.shape
    ==
    X_full.shape
)


assert (
    X_low.shape[1]
    ==
    384
)


assert (
    np.isnan(
        X_low
    ).sum()
    ==
    0
)


assert (
    np.isnan(
        X_full
    ).sum()
    ==
    0
)


# =========================================================
# 2. Verify barcode alignment
# =========================================================

low_manifest = pd.read_csv(
    LOWRES_MANIFEST_PATH
)


full_manifest = pd.read_csv(
    FULLRES_MANIFEST_PATH
)


low_barcodes = (
    low_manifest[
        "barcode"
    ]
    .astype(str)
    .tolist()
)


full_barcodes = (
    full_manifest[
        "barcode"
    ]
    .astype(str)
    .tolist()
)


if (
    low_barcodes
    !=
    full_barcodes
):

    raise ValueError(
        "Low-res and full-res barcode "
        "ordering does not match."
    )


barcodes = low_barcodes


assert (
    len(barcodes)
    ==
    X_low.shape[0]
)


print(
    "\nBarcode alignment:"
)

print(
    len(barcodes),
    "spots"
)

print(
    "Low-res / full-res ordering identical."
)


# =========================================================
# 3. Build multiscale raw representation
#
# No PCA yet.
#
# low  384
# full 384
# ----------
# concat 768
# =========================================================

X_multi = np.concatenate(
    [
        X_low,
        X_full,
    ],
    axis=1,
).astype(np.float32)


print(
    "\nMultiscale raw embedding:",
    X_multi.shape
)


FEATURE_SETS = {

    "LowRes":
        X_low,

    "FullRes":
        X_full,

    "MultiScaleConcat":
        X_multi,
}


# =========================================================
# 4. Low/full embedding similarity QC
# =========================================================

low_norm = X_low / np.clip(
    np.linalg.norm(
        X_low,
        axis=1,
        keepdims=True,
    ),
    1e-8,
    None,
)


full_norm = X_full / np.clip(
    np.linalg.norm(
        X_full,
        axis=1,
        keepdims=True,
    ),
    1e-8,
    None,
)


same_spot_cosine = np.sum(
    low_norm
    *
    full_norm,
    axis=1,
)


print(
    "\nLow/full same-spot cosine similarity:"
)

print(
    "Mean:",
    float(
        same_spot_cosine.mean()
    )
)

print(
    "Std:",
    float(
        same_spot_cosine.std()
    )
)

print(
    "Min:",
    float(
        same_spot_cosine.min()
    )
)

print(
    "Max:",
    float(
        same_spot_cosine.max()
    )
)


similarity_summary = pd.DataFrame(
    {
        "metric": [
            "mean",
            "std",
            "min",
            "max",
        ],

        "same_spot_cosine_similarity": [
            float(
                same_spot_cosine.mean()
            ),
            float(
                same_spot_cosine.std()
            ),
            float(
                same_spot_cosine.min()
            ),
            float(
                same_spot_cosine.max()
            ),
        ],
    }
)


similarity_summary.to_csv(
    OUTDIR
    / "multiscale_embedding_similarity_summary.csv",
    index=False,
)


# =========================================================
# 5. Load AnnData and split manifest
# =========================================================

print(
    "\nLoading AnnData..."
)


adata = sc.read_h5ad(
    ADATA_PATH
)


adata = adata[
    barcodes
].copy()


split_df = pd.read_csv(
    SPLIT_PATH
)


split_df[
    "barcode"
] = (
    split_df[
        "barcode"
    ]
    .astype(str)
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


# =========================================================
# 6. Expression targets
# =========================================================

Y = np.zeros(
    (
        len(barcodes),
        len(TARGET_GENES),
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


    values = adata[
        :,
        gene
    ].X


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
        .astype(np.float32)
    )


    Y[
        :,
        j
    ] = values


    print(
        f"{gene}: "
        f"mean={values.mean():.4f}, "
        f"std={values.std():.4f}"
    )


# =========================================================
# 7. Spatial graph
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
# 8. Fixed strict spatial split
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
    "\nStrict spatial split:"
)

print(
    "Train:",
    len(train_pool_idx)
)

print(
    "Test:",
    len(test_idx)
)


# =========================================================
# 9. Validation split from train blocks only
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
# 10. Model selection
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
        y_std
        <
        1e-6
    ] = 1.0


    y_fit = (
        (
            y_fit_raw
            -
            y_mean
        )
        /
        y_std
    ).astype(np.float32)


    y_val = (
        (
            y_val_raw
            -
            y_mean
        )
        /
        y_std
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
# 11. Final training
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
        y_std
        <
        1e-6
    ] = 1.0


    y_train = (
        (
            y_train_raw
            -
            y_mean
        )
        /
        y_std
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
        *
        y_std
        +
        y_mean
    )


    return pred


# =========================================================
# 12. Run benchmark
# =========================================================

all_rows = []

pca_rows = []


for (
    feature_name,
    X_raw
) in FEATURE_SETS.items():


    print(
        "\n======================================"
    )

    print(
        "Feature set:",
        feature_name
    )

    print(
        "Raw dimension:",
        X_raw.shape[1]
    )

    print(
        "======================================"
    )


    # =====================================================
    # PCA for model selection
    #
    # Fit nodes only.
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
    # Final PCA
    #
    # Entire training region only.
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
            "feature_set":
                feature_name,

            "raw_dimension":
                X_raw.shape[1],

            "pca_dimension":
                PCA_DIM,

            "train_only_pca_variance":
                explained,
        }
    )


    # =====================================================
    # Save multiscale train-only PCA features
    # =====================================================

    if (
        feature_name
        ==
        "MultiScaleConcat"
    ):

        np.save(
            OUTDIR
            / "multiscale_dinov2_train_only_pca.npy",
            X_final_all,
        )


        np.save(
            OUTDIR
            / "multiscale_dinov2_train_only_pca_components.npy",
            pca_final.components_,
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
                "feature_set":
                    feature_name,

                "model":
                    "Ridge",

                "gene":
                    gene,

                "seed":
                    np.nan,

                "best_epoch":
                    np.nan,

                **metric,
            }
        )


    # =====================================================
    # MLP / GraphSAGE
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
                feature_name,
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
                        "feature_set":
                            feature_name,

                        "model":
                            model_name,

                        "gene":
                            gene,

                        "seed":
                            seed,

                        "best_epoch":
                            best_epoch,

                        **metric,
                    }
                )


# =========================================================
# 13. Save raw metrics
# =========================================================

results = pd.DataFrame(
    all_rows
)


results.to_csv(
    OUTDIR
    / "multiscale_strict_raw_metrics.csv",
    index=False,
)


pca_summary = pd.DataFrame(
    pca_rows
)


pca_summary.to_csv(
    OUTDIR
    / "multiscale_train_only_pca_summary.csv",
    index=False,
)


# =========================================================
# 14. Summary
# =========================================================

summary_rows = []


for (
    feature_set,
    model,
    gene
), group in results.groupby(
    [
        "feature_set",
        "model",
        "gene",
    ]
):


    summary_rows.append(
        {
            "feature_set":
                feature_set,

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
                    ].std(
                        ddof=1
                    )
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
                    ].std(
                        ddof=1
                    )
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
                    ].std(
                        ddof=1
                    )
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
                    ].std(
                        ddof=1
                    )
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
    / "multiscale_strict_benchmark.csv",
    index=False,
)


# =========================================================
# 15. Compare multiscale vs each single scale
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
                "feature_set"
            ]
            ==
            "LowRes"
        ].iloc[0]


        full = sub[
            sub[
                "feature_set"
            ]
            ==
            "FullRes"
        ].iloc[0]


        multi = sub[
            sub[
                "feature_set"
            ]
            ==
            "MultiScaleConcat"
        ].iloc[0]


        best_single_pearson = max(
            low[
                "pearson_mean"
            ],
            full[
                "pearson_mean"
            ],
        )


        best_single_r2 = max(
            low[
                "r2_mean"
            ],
            full[
                "r2_mean"
            ],
        )


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

                "multiscale_pearson":
                    multi[
                        "pearson_mean"
                    ],

                "delta_multi_minus_low_pearson":
                    (
                        multi[
                            "pearson_mean"
                        ]
                        -
                        low[
                            "pearson_mean"
                        ]
                    ),

                "delta_multi_minus_full_pearson":
                    (
                        multi[
                            "pearson_mean"
                        ]
                        -
                        full[
                            "pearson_mean"
                        ]
                    ),

                "delta_multi_minus_best_single_pearson":
                    (
                        multi[
                            "pearson_mean"
                        ]
                        -
                        best_single_pearson
                    ),

                "lowres_r2":
                    low[
                        "r2_mean"
                    ],

                "fullres_r2":
                    full[
                        "r2_mean"
                    ],

                "multiscale_r2":
                    multi[
                        "r2_mean"
                    ],

                "delta_multi_minus_low_r2":
                    (
                        multi[
                            "r2_mean"
                        ]
                        -
                        low[
                            "r2_mean"
                        ]
                    ),

                "delta_multi_minus_full_r2":
                    (
                        multi[
                            "r2_mean"
                        ]
                        -
                        full[
                            "r2_mean"
                        ]
                    ),

                "delta_multi_minus_best_single_r2":
                    (
                        multi[
                            "r2_mean"
                        ]
                        -
                        best_single_r2
                    ),
            }
        )


comparison = pd.DataFrame(
    comparison_rows
)


comparison.to_csv(
    OUTDIR
    / "multiscale_vs_single_scale_comparison.csv",
    index=False,
)


# =========================================================
# 16. Paired-seed multiscale stability
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

    columns="feature_set",

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


    d_low_r = (
        group[
            (
                "pearson_r",
                "MultiScaleConcat"
            )
        ]
        -
        group[
            (
                "pearson_r",
                "LowRes"
            )
        ]
    )


    d_full_r = (
        group[
            (
                "pearson_r",
                "MultiScaleConcat"
            )
        ]
        -
        group[
            (
                "pearson_r",
                "FullRes"
            )
        ]
    )


    d_low_r2 = (
        group[
            (
                "r2",
                "MultiScaleConcat"
            )
        ]
        -
        group[
            (
                "r2",
                "LowRes"
            )
        ]
    )


    d_full_r2 = (
        group[
            (
                "r2",
                "MultiScaleConcat"
            )
        ]
        -
        group[
            (
                "r2",
                "FullRes"
            )
        ]
    )


    stability_rows.append(
        {
            "model":
                model,

            "gene":
                gene,

            "delta_multi_low_pearson_mean":
                d_low_r.mean(),

            "multi_beats_low_pearson":
                int(
                    (
                        d_low_r > 0
                    ).sum()
                ),

            "delta_multi_full_pearson_mean":
                d_full_r.mean(),

            "multi_beats_full_pearson":
                int(
                    (
                        d_full_r > 0
                    ).sum()
                ),

            "delta_multi_low_r2_mean":
                d_low_r2.mean(),

            "multi_beats_low_r2":
                int(
                    (
                        d_low_r2 > 0
                    ).sum()
                ),

            "delta_multi_full_r2_mean":
                d_full_r2.mean(),

            "multi_beats_full_r2":
                int(
                    (
                        d_full_r2 > 0
                    ).sum()
                ),

            "n_seeds":
                len(
                    d_low_r
                ),
        }
    )


stability = pd.DataFrame(
    stability_rows
)


stability.to_csv(
    OUTDIR
    / "multiscale_paired_seed_stability.csv",
    index=False,
)


# =========================================================
# 17. Print key results
# =========================================================

print(
    "\n=== Multiscale benchmark ==="
)


print(
    summary.sort_values(
        [
            "gene",
            "model",
            "feature_set",
        ]
    ).to_string(
        index=False
    )
)


print(
    "\n=== Multiscale vs single scale ==="
)


print(
    comparison.to_string(
        index=False
    )
)


print(
    "\n=== Paired seed stability ==="
)


print(
    stability.to_string(
        index=False
    )
)


# =========================================================
# 18. GraphSAGE comparison plot
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


width = 0.25


feature_order = [
    "LowRes",
    "FullRes",
    "MultiScaleConcat",
]


offsets = [
    -width,
    0,
    width,
]


fig, ax = plt.subplots(
    figsize=(11, 6)
)


for offset, feature_set in zip(
    offsets,
    feature_order,
):

    values = []

    errors = []


    for gene in TARGET_GENES:

        row = graph_df[
            (
                graph_df[
                    "feature_set"
                ]
                ==
                feature_set
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
        label=feature_set,
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
    "DINOv2 multiscale morphology\n"
    "strict spatial-block GraphSAGE"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "36_multiscale_graphsage_comparison.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 19. Multiscale gain over best single scale
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
        "delta_multi_minus_best_single_pearson"
    ],
)


ax.axhline(
    0,
    linewidth=1,
)


ax.set_ylabel(
    "Δ Pearson r\n"
    "(Multiscale - best single scale)"
)


ax.set_title(
    "Does multiscale morphology add complementary signal?"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "37_multiscale_gain_over_best_single.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# 20. PCA variance plot
# =========================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(
    pca_summary[
        "feature_set"
    ],
    pca_summary[
        "train_only_pca_variance"
    ],
)


ax.set_ylabel(
    "Variance explained by 50 PCs"
)


ax.set_title(
    "Train-only PCA compression"
)


plt.tight_layout()


plt.savefig(
    FIGDIR
    / "38_multiscale_pca_variance.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close()


# =========================================================
# Final report
# =========================================================

print(
    "\nSaved:"
)

print(
    "outputs/multiscale_embedding_similarity_summary.csv"
)

print(
    "outputs/multiscale_dinov2_train_only_pca.npy"
)

print(
    "outputs/multiscale_train_only_pca_summary.csv"
)

print(
    "outputs/multiscale_strict_raw_metrics.csv"
)

print(
    "outputs/multiscale_strict_benchmark.csv"
)

print(
    "outputs/multiscale_vs_single_scale_comparison.csv"
)

print(
    "outputs/multiscale_paired_seed_stability.csv"
)

print(
    "figures/36_multiscale_graphsage_comparison.png"
)

print(
    "figures/37_multiscale_gain_over_best_single.png"
)

print(
    "figures/38_multiscale_pca_variance.png"
)


print(
    "\nStrict multiscale DINOv2 fusion experiment complete."
)
