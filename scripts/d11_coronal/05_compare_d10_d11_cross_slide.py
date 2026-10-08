from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

BASE = Path(
    "/media/sophie/Data2T/Biomedical AI  skill"
)

D10 = (
    BASE
    / "D10_mouse_brain_external_validation"
)

D11 = (
    BASE
    / "D11_mouse_brain_coronal_validation"
)

D10_OUT = D10 / "outputs"
D11_OUT = D11 / "outputs"

FIG = D11 / "figures"

FIG.mkdir(
    exist_ok=True
)


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

D10_RAW = (
    D10_OUT
    / "unsupervised_alignment_raw_metrics.csv"
)

D10_SUMMARY = (
    D10_OUT
    / "unsupervised_alignment_summary.csv"
)

D10_BEST = (
    D10_OUT
    / "unsupervised_alignment_best_models.csv"
)


D11_RAW = (
    D11_OUT
    / "coronal_unsupervised_alignment_raw_metrics.csv"
)

D11_SUMMARY = (
    D11_OUT
    / "coronal_unsupervised_alignment_summary.csv"
)

D11_BEST = (
    D11_OUT
    / "coronal_unsupervised_alignment_best_models.csv"
)


TARGET_GENES = [
    "Camk2n1",
    "Slc17a7",
    "Plp1",
    "Cldn11",
]


ALIGNMENTS = [
    "None",
    "ZScore",
    "CORAL",
]


MODELS = [
    "LowExpert_GraphSAGE",
    "FullExpert_GraphSAGE",
    "SeparatePCA50x2_GraphSAGE",
    "FrozenStaticMoE_GraphSAGE",
]


# ============================================================
# Check files
# ============================================================

required = [
    D10_RAW,
    D10_SUMMARY,
    D11_RAW,
    D11_SUMMARY,
]


print(
    "Checking inputs..."
)


for p in required:

    print(
        p,
        "FOUND"
        if p.exists()
        else "MISSING"
    )


    if not p.exists():

        raise FileNotFoundError(
            str(p)
        )


# ============================================================
# Load
# ============================================================

d10_raw = pd.read_csv(
    D10_RAW
)

d11_raw = pd.read_csv(
    D11_RAW
)


d10_summary = pd.read_csv(
    D10_SUMMARY
)

d11_summary = pd.read_csv(
    D11_SUMMARY
)


# ============================================================
# Normalize alignment labels
#
# Pandas may parse the literal string "None" from CSV as NaN.
# Restore it explicitly so strict zero-shot rows are retained.
# ============================================================

for df in [
    d10_raw,
    d11_raw,
    d10_summary,
    d11_summary,
]:
    df["alignment"] = (
        df["alignment"]
        .fillna("None")
        .astype(str)
    )


print(
    "\nD10 raw:",
    d10_raw.shape
)

print(
    "D11 raw:",
    d11_raw.shape
)

print(
    "D10 summary:",
    d10_summary.shape
)

print(
    "D11 summary:",
    d11_summary.shape
)


# ============================================================
# Basic schema QC
# ============================================================

raw_required_cols = [
    "alignment",
    "model",
    "gene",
    "seed",
    "pearson_r",
    "spearman_rho",
    "r2",
    "rmse",
]


summary_required_cols = [
    "alignment",
    "model",
    "gene",
    "pearson_mean",
    "pearson_std",
    "spearman_mean",
    "spearman_std",
    "r2_mean",
    "r2_std",
    "rmse_mean",
    "rmse_std",
]


for name, df in [
    (
        "D10 raw",
        d10_raw
    ),
    (
        "D11 raw",
        d11_raw
    ),
]:

    missing = [
        c
        for c in raw_required_cols
        if c not in df.columns
    ]


    if missing:

        raise KeyError(
            f"{name} missing columns: "
            f"{missing}"
        )


for name, df in [
    (
        "D10 summary",
        d10_summary
    ),
    (
        "D11 summary",
        d11_summary
    ),
]:

    missing = [
        c
        for c in summary_required_cols
        if c not in df.columns
    ]


    if missing:

        raise KeyError(
            f"{name} missing columns: "
            f"{missing}"
        )


# ============================================================
# Dataset labels
# ============================================================

d10_raw[
    "target_dataset"
] = (
    "D10_sagittal"
)

d11_raw[
    "target_dataset"
] = (
    "D11_coronal"
)


d10_summary[
    "target_dataset"
] = (
    "D10_sagittal"
)

d11_summary[
    "target_dataset"
] = (
    "D11_coronal"
)


# ============================================================
# 1. Combined long-format table
# ============================================================

combined_summary = pd.concat(
    [
        d10_summary,
        d11_summary,
    ],
    axis=0,
    ignore_index=True,
)


combined_summary = combined_summary[
    [
        "target_dataset",
        "alignment",
        "model",
        "gene",
        "pearson_mean",
        "pearson_std",
        "spearman_mean",
        "spearman_std",
        "r2_mean",
        "r2_std",
        "rmse_mean",
        "rmse_std",
    ]
]


combined_summary.to_csv(
    D11_OUT
    / "cross_slide_d10_d11_all_results.csv",
    index=False
)


# ============================================================
# 2. Strict zero-shot only
#
# alignment = None
# ============================================================

strict = combined_summary[
    combined_summary[
        "alignment"
    ]
    ==
    "None"
].copy()


strict.to_csv(
    D11_OUT
    / "cross_slide_d10_d11_strict_zero_shot_all_models.csv",
    index=False
)


# ============================================================
# 3. Best STRICT zero-shot model per gene / target
# ============================================================

strict_best_rows = []


for target in [
    "D10_sagittal",
    "D11_coronal",
]:

    for gene in TARGET_GENES:

        g = strict[
            (
                strict[
                    "target_dataset"
                ]
                ==
                target
            )
            &
            (
                strict[
                    "gene"
                ]
                ==
                gene
            )
        ].copy()


        if len(g) == 0:

            continue


        best_r = g.loc[
            g[
                "pearson_mean"
            ].idxmax()
        ]


        best_r2 = g.loc[
            g[
                "r2_mean"
            ].idxmax()
        ]


        strict_best_rows.append(
            {
                "target_dataset":
                    target,

                "gene":
                    gene,

                "best_pearson_model":
                    best_r[
                        "model"
                    ],

                "best_pearson":
                    float(
                        best_r[
                            "pearson_mean"
                        ]
                    ),

                "best_pearson_std":
                    float(
                        best_r[
                            "pearson_std"
                        ]
                    ),

                "best_pearson_r2":
                    float(
                        best_r[
                            "r2_mean"
                        ]
                    ),

                "best_r2_model":
                    best_r2[
                        "model"
                    ],

                "best_r2":
                    float(
                        best_r2[
                            "r2_mean"
                        ]
                    ),

                "best_r2_pearson":
                    float(
                        best_r2[
                            "pearson_mean"
                        ]
                    ),
            }
        )


strict_best = pd.DataFrame(
    strict_best_rows
)


strict_best.to_csv(
    D11_OUT
    / "cross_slide_d10_d11_strict_zero_shot_best.csv",
    index=False
)


# ============================================================
# 4. Best result within each alignment
#
# Important:
# This is descriptive benchmarking only.
# ============================================================

alignment_best_rows = []


for target in [
    "D10_sagittal",
    "D11_coronal",
]:

    for alignment in ALIGNMENTS:

        for gene in TARGET_GENES:

            g = combined_summary[
                (
                    combined_summary[
                        "target_dataset"
                    ]
                    ==
                    target
                )
                &
                (
                    combined_summary[
                        "alignment"
                    ]
                    ==
                    alignment
                )
                &
                (
                    combined_summary[
                        "gene"
                    ]
                    ==
                    gene
                )
            ].copy()


            if len(g) == 0:

                continue


            best = g.loc[
                g[
                    "pearson_mean"
                ].idxmax()
            ]


            alignment_best_rows.append(
                {
                    "target_dataset":
                        target,

                    "alignment":
                        alignment,

                    "gene":
                        gene,

                    "best_model":
                        best[
                            "model"
                        ],

                    "pearson_mean":
                        float(
                            best[
                                "pearson_mean"
                            ]
                        ),

                    "pearson_std":
                        float(
                            best[
                                "pearson_std"
                            ]
                        ),

                    "spearman_mean":
                        float(
                            best[
                                "spearman_mean"
                            ]
                        ),

                    "r2_mean":
                        float(
                            best[
                                "r2_mean"
                            ]
                        ),

                    "rmse_mean":
                        float(
                            best[
                                "rmse_mean"
                            ]
                        ),
                }
            )


alignment_best = pd.DataFrame(
    alignment_best_rows
)


alignment_best.to_csv(
    D11_OUT
    / "cross_slide_d10_d11_best_by_alignment.csv",
    index=False
)


# ============================================================
# 5. Overall descriptive best per gene
#
# This searches across:
# None / ZScore / CORAL
# and all four architectures.
#
# DO NOT interpret this as a pre-specified deployed model.
# ============================================================

overall_best_rows = []


for target in [
    "D10_sagittal",
    "D11_coronal",
]:

    for gene in TARGET_GENES:

        g = combined_summary[
            (
                combined_summary[
                    "target_dataset"
                ]
                ==
                target
            )
            &
            (
                combined_summary[
                    "gene"
                ]
                ==
                gene
            )
        ].copy()


        if len(g) == 0:

            continue


        best_r = g.loc[
            g[
                "pearson_mean"
            ].idxmax()
        ]


        best_r2 = g.loc[
            g[
                "r2_mean"
            ].idxmax()
        ]


        overall_best_rows.append(
            {
                "target_dataset":
                    target,

                "gene":
                    gene,

                "best_pearson_alignment":
                    best_r[
                        "alignment"
                    ],

                "best_pearson_model":
                    best_r[
                        "model"
                    ],

                "best_pearson":
                    float(
                        best_r[
                            "pearson_mean"
                        ]
                    ),

                "best_pearson_std":
                    float(
                        best_r[
                            "pearson_std"
                        ]
                    ),

                "best_r2_alignment":
                    best_r2[
                        "alignment"
                    ],

                "best_r2_model":
                    best_r2[
                        "model"
                    ],

                "best_r2":
                    float(
                        best_r2[
                            "r2_mean"
                        ]
                    ),
            }
        )


overall_best = pd.DataFrame(
    overall_best_rows
)


overall_best.to_csv(
    D11_OUT
    / "cross_slide_d10_d11_overall_descriptive_best.csv",
    index=False
)


# ============================================================
# 6. Descriptive D11 - D10 best delta
# ============================================================

best_d10 = (
    overall_best[
        overall_best[
            "target_dataset"
        ]
        ==
        "D10_sagittal"
    ]
    .set_index(
        "gene"
    )
)


best_d11 = (
    overall_best[
        overall_best[
            "target_dataset"
        ]
        ==
        "D11_coronal"
    ]
    .set_index(
        "gene"
    )
)


best_delta_rows = []


for gene in TARGET_GENES:

    if (
        gene not in best_d10.index
        or
        gene not in best_d11.index
    ):

        continue


    r10 = float(
        best_d10.loc[
            gene,
            "best_pearson"
        ]
    )


    r11 = float(
        best_d11.loc[
            gene,
            "best_pearson"
        ]
    )


    best_delta_rows.append(
        {
            "gene":
                gene,

            "d10_sagittal_best_pearson":
                r10,

            "d11_coronal_best_pearson":
                r11,

            "delta_d11_minus_d10":
                r11 - r10,

            "d10_alignment":
                best_d10.loc[
                    gene,
                    "best_pearson_alignment"
                ],

            "d10_model":
                best_d10.loc[
                    gene,
                    "best_pearson_model"
                ],

            "d11_alignment":
                best_d11.loc[
                    gene,
                    "best_pearson_alignment"
                ],

            "d11_model":
                best_d11.loc[
                    gene,
                    "best_pearson_model"
                ],
        }
    )


best_delta = pd.DataFrame(
    best_delta_rows
)


best_delta.to_csv(
    D11_OUT
    / "cross_slide_d10_d11_descriptive_best_delta.csv",
    index=False
)


# ============================================================
# 7. STRICT zero-shot descriptive delta
# ============================================================

strict_d10 = (
    strict_best[
        strict_best[
            "target_dataset"
        ]
        ==
        "D10_sagittal"
    ]
    .set_index(
        "gene"
    )
)


strict_d11 = (
    strict_best[
        strict_best[
            "target_dataset"
        ]
        ==
        "D11_coronal"
    ]
    .set_index(
        "gene"
    )
)


strict_delta_rows = []


for gene in TARGET_GENES:

    if (
        gene not in strict_d10.index
        or
        gene not in strict_d11.index
    ):

        continue


    r10 = float(
        strict_d10.loc[
            gene,
            "best_pearson"
        ]
    )


    r11 = float(
        strict_d11.loc[
            gene,
            "best_pearson"
        ]
    )


    strict_delta_rows.append(
        {
            "gene":
                gene,

            "d10_sagittal_strict_best_r":
                r10,

            "d11_coronal_strict_best_r":
                r11,

            "delta_r":
                r11 - r10,

            "d10_best_model":
                strict_d10.loc[
                    gene,
                    "best_pearson_model"
                ],

            "d11_best_model":
                strict_d11.loc[
                    gene,
                    "best_pearson_model"
                ],

            "d10_best_r2":
                float(
                    strict_d10.loc[
                        gene,
                        "best_r2"
                    ]
                ),

            "d11_best_r2":
                float(
                    strict_d11.loc[
                        gene,
                        "best_r2"
                    ]
                ),
        }
    )


strict_delta = pd.DataFrame(
    strict_delta_rows
)


strict_delta.to_csv(
    D11_OUT
    / "cross_slide_d10_d11_strict_best_delta.csv",
    index=False
)


# ============================================================
# 8. Matched model + alignment + seed comparison
#
# This is the strongest comparison.
#
# Same:
#   model
#   alignment
#   gene
#   seed
#
# Only target changes:
#   sagittal D10
#   coronal D11
# ============================================================

match_cols = [
    "alignment",
    "model",
    "gene",
    "seed",
]


paired = d11_raw.merge(
    d10_raw[
        [
            "alignment",
            "model",
            "gene",
            "seed",
            "pearson_r",
            "spearman_rho",
            "r2",
            "rmse",
        ]
    ],
    on=match_cols,
    suffixes=(
        "_d11_coronal",
        "_d10_sagittal"
    ),
    validate="one_to_one",
)


paired[
    "delta_pearson_d11_minus_d10"
] = (
    paired[
        "pearson_r_d11_coronal"
    ]
    -
    paired[
        "pearson_r_d10_sagittal"
    ]
)


paired[
    "delta_spearman_d11_minus_d10"
] = (
    paired[
        "spearman_rho_d11_coronal"
    ]
    -
    paired[
        "spearman_rho_d10_sagittal"
    ]
)


paired[
    "delta_r2_d11_minus_d10"
] = (
    paired[
        "r2_d11_coronal"
    ]
    -
    paired[
        "r2_d10_sagittal"
    ]
)


paired[
    "delta_rmse_d11_minus_d10"
] = (
    paired[
        "rmse_d11_coronal"
    ]
    -
    paired[
        "rmse_d10_sagittal"
    ]
)


paired.to_csv(
    D11_OUT
    / "cross_slide_d10_d11_matched_seed_deltas.csv",
    index=False
)


# ============================================================
# 9. Aggregate matched deltas
# ============================================================

matched_rows = []


group_cols = [
    "alignment",
    "model",
    "gene",
]


for keys, g in paired.groupby(
    group_cols
):

    alignment, model, gene = keys


    dr = g[
        "delta_pearson_d11_minus_d10"
    ]


    ds = g[
        "delta_spearman_d11_minus_d10"
    ]


    dr2 = g[
        "delta_r2_d11_minus_d10"
    ]


    drmse = g[
        "delta_rmse_d11_minus_d10"
    ]


    matched_rows.append(
        {
            "alignment":
                alignment,

            "model":
                model,

            "gene":
                gene,

            "d10_pearson_mean":
                float(
                    g[
                        "pearson_r_d10_sagittal"
                    ].mean()
                ),

            "d11_pearson_mean":
                float(
                    g[
                        "pearson_r_d11_coronal"
                    ].mean()
                ),

            "delta_pearson_mean":
                float(
                    dr.mean()
                ),

            "delta_pearson_std":
                float(
                    dr.std()
                ),

            "d11_pearson_wins":
                int(
                    (
                        dr > 0
                    ).sum()
                ),

            "d10_spearman_mean":
                float(
                    g[
                        "spearman_rho_d10_sagittal"
                    ].mean()
                ),

            "d11_spearman_mean":
                float(
                    g[
                        "spearman_rho_d11_coronal"
                    ].mean()
                ),

            "delta_spearman_mean":
                float(
                    ds.mean()
                ),

            "d10_r2_mean":
                float(
                    g[
                        "r2_d10_sagittal"
                    ].mean()
                ),

            "d11_r2_mean":
                float(
                    g[
                        "r2_d11_coronal"
                    ].mean()
                ),

            "delta_r2_mean":
                float(
                    dr2.mean()
                ),

            "d11_r2_wins":
                int(
                    (
                        dr2 > 0
                    ).sum()
                ),

            "d10_rmse_mean":
                float(
                    g[
                        "rmse_d10_sagittal"
                    ].mean()
                ),

            "d11_rmse_mean":
                float(
                    g[
                        "rmse_d11_coronal"
                    ].mean()
                ),

            "delta_rmse_mean":
                float(
                    drmse.mean()
                ),

            "d11_rmse_wins":
                int(
                    (
                        drmse < 0
                    ).sum()
                ),

            "n_seeds":
                int(
                    len(g)
                ),
        }
    )


matched_summary = pd.DataFrame(
    matched_rows
)


matched_summary.to_csv(
    D11_OUT
    / "cross_slide_d10_d11_matched_method_summary.csv",
    index=False
)


# ============================================================
# 10. Gene-level matched comparison
#
# Average across ALL fixed model/alignment/seed combinations.
#
# This avoids choosing a target-specific winner.
# ============================================================

gene_rows = []


for gene in TARGET_GENES:

    g = paired[
        paired[
            "gene"
        ]
        ==
        gene
    ]


    if len(g) == 0:

        continue


    dr = g[
        "delta_pearson_d11_minus_d10"
    ]


    dr2 = g[
        "delta_r2_d11_minus_d10"
    ]


    gene_rows.append(
        {
            "gene":
                gene,

            "n_matched_runs":
                int(
                    len(g)
                ),

            "d10_pearson_mean_all_methods":
                float(
                    g[
                        "pearson_r_d10_sagittal"
                    ].mean()
                ),

            "d11_pearson_mean_all_methods":
                float(
                    g[
                        "pearson_r_d11_coronal"
                    ].mean()
                ),

            "mean_delta_pearson":
                float(
                    dr.mean()
                ),

            "median_delta_pearson":
                float(
                    dr.median()
                ),

            "fraction_d11_better_pearson":
                float(
                    (
                        dr > 0
                    ).mean()
                ),

            "d10_r2_mean_all_methods":
                float(
                    g[
                        "r2_d10_sagittal"
                    ].mean()
                ),

            "d11_r2_mean_all_methods":
                float(
                    g[
                        "r2_d11_coronal"
                    ].mean()
                ),

            "mean_delta_r2":
                float(
                    dr2.mean()
                ),

            "fraction_d11_better_r2":
                float(
                    (
                        dr2 > 0
                    ).mean()
                ),
        }
    )


gene_comparison = pd.DataFrame(
    gene_rows
)


gene_comparison.to_csv(
    D11_OUT
    / "cross_slide_d10_d11_gene_level_matched_comparison.csv",
    index=False
)


# ============================================================
# 11. Alignment-level matched comparison
# ============================================================

alignment_rows = []


for alignment in ALIGNMENTS:

    g = paired[
        paired[
            "alignment"
        ]
        ==
        alignment
    ]


    if len(g) == 0:

        continue


    dr = g[
        "delta_pearson_d11_minus_d10"
    ]


    alignment_rows.append(
        {
            "alignment":
                alignment,

            "n_matched_runs":
                int(
                    len(g)
                ),

            "d10_pearson_mean":
                float(
                    g[
                        "pearson_r_d10_sagittal"
                    ].mean()
                ),

            "d11_pearson_mean":
                float(
                    g[
                        "pearson_r_d11_coronal"
                    ].mean()
                ),

            "mean_delta_pearson":
                float(
                    dr.mean()
                ),

            "median_delta_pearson":
                float(
                    dr.median()
                ),

            "fraction_d11_better":
                float(
                    (
                        dr > 0
                    ).mean()
                ),
        }
    )


alignment_comparison = pd.DataFrame(
    alignment_rows
)


alignment_comparison.to_csv(
    D11_OUT
    / "cross_slide_d10_d11_alignment_level_comparison.csv",
    index=False
)


# ============================================================
# 12. Figure — strict zero-shot best Pearson
# ============================================================

plot_df = strict_delta.set_index(
    "gene"
).loc[
    TARGET_GENES
].reset_index()


x = np.arange(
    len(
        TARGET_GENES
    )
)

width = 0.36


fig, ax = plt.subplots(
    figsize=(
        9,
        5.5
    )
)


ax.bar(
    x - width / 2,
    plot_df[
        "d10_sagittal_strict_best_r"
    ],
    width,
    label="D10 sagittal"
)


ax.bar(
    x + width / 2,
    plot_df[
        "d11_coronal_strict_best_r"
    ],
    width,
    label="D11 coronal"
)


ax.axhline(
    0,
    linewidth=1
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
    "D9 cross-slide strict zero-shot transfer"
)


ax.legend()


for i, row in plot_df.iterrows():

    ax.text(
        i - width / 2,
        row[
            "d10_sagittal_strict_best_r"
        ]
        +
        0.015,
        f"{row['d10_sagittal_strict_best_r']:.2f}",
        ha="center",
        fontsize=8
    )


    ax.text(
        i + width / 2,
        row[
            "d11_coronal_strict_best_r"
        ]
        +
        0.015,
        f"{row['d11_coronal_strict_best_r']:.2f}",
        ha="center",
        fontsize=8
    )


plt.tight_layout()


plt.savefig(
    FIG
    / "06_d10_sagittal_vs_d11_coronal_strict_zero_shot.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 13. Figure — overall descriptive best
# ============================================================

best_plot = (
    best_delta
    .set_index(
        "gene"
    )
    .loc[
        TARGET_GENES
    ]
    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(
        9,
        5.5
    )
)


ax.bar(
    x - width / 2,
    best_plot[
        "d10_sagittal_best_pearson"
    ],
    width,
    label="D10 sagittal"
)


ax.bar(
    x + width / 2,
    best_plot[
        "d11_coronal_best_pearson"
    ],
    width,
    label="D11 coronal"
)


ax.axhline(
    0,
    linewidth=1
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
    "D9 cross-slide transfer: descriptive best"
)


ax.legend()


for i, row in best_plot.iterrows():

    ax.text(
        i - width / 2,
        row[
            "d10_sagittal_best_pearson"
        ]
        +
        0.015,
        f"{row['d10_sagittal_best_pearson']:.2f}",
        ha="center",
        fontsize=8
    )


    ax.text(
        i + width / 2,
        row[
            "d11_coronal_best_pearson"
        ]
        +
        0.015,
        f"{row['d11_coronal_best_pearson']:.2f}",
        ha="center",
        fontsize=8
    )


plt.tight_layout()


plt.savefig(
    FIG
    / "07_d10_sagittal_vs_d11_coronal_descriptive_best.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 14. Figure — matched mean delta by gene
# ============================================================

gene_plot = (
    gene_comparison
    .set_index(
        "gene"
    )
    .loc[
        TARGET_GENES
    ]
    .reset_index()
)


fig, ax = plt.subplots(
    figsize=(
        8,
        5
    )
)


bars = ax.bar(
    gene_plot[
        "gene"
    ],
    gene_plot[
        "mean_delta_pearson"
    ]
)


ax.axhline(
    0,
    linewidth=1
)


ax.set_ylabel(
    "Mean Δ Pearson r\n"
    "(D11 coronal - D10 sagittal)"
)


ax.set_title(
    "Matched cross-slide transfer advantage"
)


for bar, value in zip(
    bars,
    gene_plot[
        "mean_delta_pearson"
    ]
):

    ax.text(
        bar.get_x()
        +
        bar.get_width() / 2,
        value
        +
        (
            0.01
            if value >= 0
            else -0.02
        ),
        f"{value:.3f}",
        ha="center",
        va=(
            "bottom"
            if value >= 0
            else "top"
        ),
        fontsize=9
    )


plt.tight_layout()


plt.savefig(
    FIG
    / "08_d10_d11_matched_delta_pearson_by_gene.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 15. Console report
# ============================================================

print(
    "\n============================================"
)

print(
    "D10 SAGITTAL vs D11 CORONAL CROSS-SLIDE"
)

print(
    "============================================"
)


print(
    "\nSTRICT ZERO-SHOT BEST:"
)


print(
    strict_delta.to_string(
        index=False
    )
)


print(
    "\nOVERALL DESCRIPTIVE BEST:"
)


print(
    best_delta.to_string(
        index=False
    )
)


print(
    "\nGENE-LEVEL MATCHED COMPARISON:"
)


print(
    gene_comparison.to_string(
        index=False
    )
)


print(
    "\nALIGNMENT-LEVEL MATCHED COMPARISON:"
)


print(
    alignment_comparison.to_string(
        index=False
    )
)


print(
    "\nInterpretation rule:"
)

print(
    "Positive delta means D9 -> D11 coronal "
    "outperformed D9 -> D10 sagittal."
)

print(
    "The matched model/alignment/seed comparison "
    "is the primary analysis."
)

print(
    "The per-target 'best' tables are descriptive "
    "and should not be treated as a pre-specified "
    "model-selection result."
)


print(
    "\nSaved outputs:"
)


for name in [
    "cross_slide_d10_d11_all_results.csv",
    "cross_slide_d10_d11_strict_zero_shot_all_models.csv",
    "cross_slide_d10_d11_strict_zero_shot_best.csv",
    "cross_slide_d10_d11_best_by_alignment.csv",
    "cross_slide_d10_d11_overall_descriptive_best.csv",
    "cross_slide_d10_d11_descriptive_best_delta.csv",
    "cross_slide_d10_d11_strict_best_delta.csv",
    "cross_slide_d10_d11_matched_seed_deltas.csv",
    "cross_slide_d10_d11_matched_method_summary.csv",
    "cross_slide_d10_d11_gene_level_matched_comparison.csv",
    "cross_slide_d10_d11_alignment_level_comparison.csv",
]:

    print(
        D11_OUT
        /
        name
    )


print(
    "\nSaved figures:"
)


for name in [
    "06_d10_sagittal_vs_d11_coronal_strict_zero_shot.png",
    "07_d10_sagittal_vs_d11_coronal_descriptive_best.png",
    "08_d10_d11_matched_delta_pearson_by_gene.png",
]:

    print(
        FIG
        /
        name
    )
