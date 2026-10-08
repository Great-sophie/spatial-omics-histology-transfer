import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA


OUTDIR = Path("outputs")
FIGDIR = Path("figures")

OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

EMBEDDING_FILE = OUTDIR / "resnet50_image_embeddings.npy"
MANIFEST_FILE = OUTDIR / "spot_image_embedding_manifest.csv"


print("Loading saved ResNet50 embeddings...")

embeddings = np.load(EMBEDDING_FILE)
manifest = pd.read_csv(MANIFEST_FILE)

print("Embeddings:", embeddings.shape)
print("Manifest:", manifest.shape)

assert embeddings.shape[0] == len(manifest)
assert np.isnan(embeddings).sum() == 0


# =========================================================
# 1. Fast PCA
# =========================================================

print("\nRunning randomized PCA...")

pca = PCA(
    n_components=50,
    svd_solver="randomized",
    random_state=42,
)

embedding_pca = pca.fit_transform(embeddings)

print("PCA shape:", embedding_pca.shape)
print("PCA NaNs:", np.isnan(embedding_pca).sum())

print("\nFirst 10 explained variance ratios:")
print(pca.explained_variance_ratio_[:10])

print(
    "\nTotal explained variance (50 PCs):",
    pca.explained_variance_ratio_.sum()
)


# =========================================================
# 2. Save PCA array
# =========================================================

np.save(
    OUTDIR / "resnet50_image_embeddings_pca.npy",
    embedding_pca,
)


# =========================================================
# 3. Save PCA metadata table
# =========================================================

keep_cols = [
    "barcode",
    "cluster",
    "leiden",
    "fullres_x",
    "fullres_y",
    "image_x",
    "image_y",
]

keep_cols = [
    c for c in keep_cols
    if c in manifest.columns
]

pca_df = manifest[keep_cols].copy()

for i in range(10):
    pca_df[f"image_PC{i+1}"] = embedding_pca[:, i]

pca_df.to_csv(
    OUTDIR / "image_embedding_pca.csv",
    index=False,
)


# =========================================================
# 4. PC1 vs PC2 plot
# =========================================================

if "cluster" in manifest.columns:

    clusters = pd.Categorical(manifest["cluster"])

    plt.figure(figsize=(8, 7))

    plt.scatter(
        embedding_pca[:, 0],
        embedding_pca[:, 1],
        c=clusters.codes,
        s=10,
        alpha=0.75,
    )

    plt.xlabel("Image embedding PC1")
    plt.ylabel("Image embedding PC2")
    plt.title("ResNet50 H&E morphology embeddings")

    plt.tight_layout()

    plt.savefig(
        FIGDIR / "03_resnet50_embedding_pca.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =========================================================
# 5. Spatial PC1
# =========================================================

x = manifest["image_x"].values
y = manifest["image_y"].values

plt.figure(figsize=(8, 8))

sc = plt.scatter(
    x,
    y,
    c=embedding_pca[:, 0],
    s=10,
)

plt.gca().invert_yaxis()
plt.gca().set_aspect("equal")

plt.xlabel("Spatial x")
plt.ylabel("Spatial y")
plt.title("Spatial distribution of morphology PC1")

plt.colorbar(sc, label="Image PC1")

plt.tight_layout()

plt.savefig(
    FIGDIR / "04_morphology_PC1_spatial.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# 6. Spatial PC2
# =========================================================

plt.figure(figsize=(8, 8))

sc = plt.scatter(
    x,
    y,
    c=embedding_pca[:, 1],
    s=10,
)

plt.gca().invert_yaxis()
plt.gca().set_aspect("equal")

plt.xlabel("Spatial x")
plt.ylabel("Spatial y")
plt.title("Spatial distribution of morphology PC2")

plt.colorbar(sc, label="Image PC2")

plt.tight_layout()

plt.savefig(
    FIGDIR / "05_morphology_PC2_spatial.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================================================
# 7. Summary
# =========================================================

summary = pd.DataFrame(
    {
        "metric": [
            "n_patches",
            "embedding_model",
            "embedding_dimension",
            "pca_dimension",
            "pca_solver",
            "embedding_nan_count",
            "pca_nan_count",
            "pca_explained_variance_total",
        ],
        "value": [
            embeddings.shape[0],
            "ResNet50_ImageNet",
            embeddings.shape[1],
            embedding_pca.shape[1],
            "randomized",
            int(np.isnan(embeddings).sum()),
            int(np.isnan(embedding_pca).sum()),
            float(pca.explained_variance_ratio_.sum()),
        ],
    }
)

summary.to_csv(
    OUTDIR / "image_embedding_summary.csv",
    index=False,
)


print("\nSaved:")
print("outputs/resnet50_image_embeddings_pca.npy")
print("outputs/image_embedding_pca.csv")
print("outputs/image_embedding_summary.csv")
print("figures/03_resnet50_embedding_pca.png")
print("figures/04_morphology_PC1_spatial.png")
print("figures/05_morphology_PC2_spatial.png")

print("\nPCA stage complete.")
