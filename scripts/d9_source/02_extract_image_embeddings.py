import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.decomposition import PCA


# =========================================================
# Configuration
# =========================================================

MANIFEST = Path("outputs/spot_patch_manifest.csv")

OUTDIR = Path("outputs")
FIGDIR = Path("figures")

OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
NUM_WORKERS = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# 1. Load manifest
# =========================================================

manifest = pd.read_csv(MANIFEST)

print("Number of patches:", len(manifest))
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# 2. Load pretrained ResNet50
# =========================================================

weights = ResNet50_Weights.DEFAULT

transform = weights.transforms()

model = resnet50(weights=weights)

# Remove classification head.
# ResNet50 output becomes a 2048-D morphology embedding.
model.fc = torch.nn.Identity()

model = model.to(DEVICE)
model.eval()

print("\nModel:")
print("ResNet50 pretrained on ImageNet")
print("Embedding dimension: 2048")


# =========================================================
# 3. Dataset
# =========================================================

class PatchDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        patch_path = row["patch_path"]

        image = Image.open(
            patch_path
        ).convert("RGB")

        image = self.transform(image)

        return image, idx


dataset = PatchDataset(
    manifest,
    transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)


# =========================================================
# 4. Extract morphology embeddings
# =========================================================

all_embeddings = []
all_indices = []

print("\nExtracting image embeddings...")

with torch.no_grad():

    for batch_idx, (images, indices) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        embeddings = model(images)

        embeddings = embeddings.cpu().numpy()

        all_embeddings.append(
            embeddings
        )

        all_indices.append(
            indices.numpy()
        )

        if batch_idx % 10 == 0:

            done = min(
                (batch_idx + 1) * BATCH_SIZE,
                len(dataset),
            )

            print(
                f"{done}/{len(dataset)} patches processed"
            )


embeddings = np.concatenate(
    all_embeddings,
    axis=0,
)

indices = np.concatenate(
    all_indices,
    axis=0,
)


# =========================================================
# 5. Sanity checks
# =========================================================

assert embeddings.shape[0] == len(manifest)

print("\nEmbedding matrix:")
print(embeddings.shape)

print(
    "Embedding mean:",
    float(embeddings.mean())
)

print(
    "Embedding std:",
    float(embeddings.std())
)

print(
    "NaN count:",
    int(np.isnan(embeddings).sum())
)


# =========================================================
# 6. Save raw embeddings
# =========================================================

np.save(
    OUTDIR / "resnet50_image_embeddings.npy",
    embeddings,
)

manifest_out = manifest.copy()

manifest_out["embedding_index"] = np.arange(
    len(manifest_out)
)

manifest_out.to_csv(
    OUTDIR /
    "spot_image_embedding_manifest.csv",
    index=False,
)


# =========================================================
# 7. PCA of morphology embeddings
# =========================================================

n_components = min(
    50,
    embeddings.shape[0],
    embeddings.shape[1],
)

pca = PCA(
    n_components=n_components,
    random_state=42,
)

embedding_pca = pca.fit_transform(
    embeddings
)

print("\nImage embedding PCA:")
print(embedding_pca.shape)

print(
    "\nExplained variance ratio "
    "for first 10 PCs:"
)

print(
    pca.explained_variance_ratio_[:10]
)

print(
    "\nTotal explained variance "
    "for 50 PCs:"
)

print(
    pca.explained_variance_ratio_.sum()
)


# =========================================================
# 8. Save PCA representation
# =========================================================

np.save(
    OUTDIR /
    "resnet50_image_embeddings_pca.npy",
    embedding_pca,
)


# =========================================================
# 9. Create PCA metadata table
# =========================================================

columns_to_keep = [
    "barcode",
    "cluster",
    "leiden",
    "fullres_x",
    "fullres_y",
    "image_x",
    "image_y",
]

columns_to_keep = [
    c for c in columns_to_keep
    if c in manifest.columns
]

pca_df = manifest[
    columns_to_keep
].copy()

for i in range(
    min(10, embedding_pca.shape[1])
):

    pca_df[
        f"image_PC{i + 1}"
    ] = embedding_pca[:, i]


pca_df.to_csv(
    OUTDIR /
    "image_embedding_pca.csv",
    index=False,
)


# =========================================================
# 10. PCA scatter colored by anatomical cluster
# =========================================================

if "cluster" in manifest.columns:

    clusters = pd.Categorical(
        manifest["cluster"]
    )

    cluster_codes = clusters.codes

    plt.figure(
        figsize=(8, 7)
    )

    scatter = plt.scatter(
        embedding_pca[:, 0],
        embedding_pca[:, 1],
        c=cluster_codes,
        s=10,
        alpha=0.75,
    )

    plt.xlabel(
        "Image embedding PC1"
    )

    plt.ylabel(
        "Image embedding PC2"
    )

    plt.title(
        "ResNet50 H&E morphology embeddings"
    )

    plt.tight_layout()

    plt.savefig(
        FIGDIR /
        "03_resnet50_embedding_pca.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =========================================================
# 11. Spatial visualization of morphology PC1
# =========================================================

if (
    "image_x" in manifest.columns
    and "image_y" in manifest.columns
):

    coords_x = manifest[
        "image_x"
    ].values

    coords_y = manifest[
        "image_y"
    ].values

    plt.figure(
        figsize=(8, 8)
    )

    scatter = plt.scatter(
        coords_x,
        coords_y,
        c=embedding_pca[:, 0],
        s=10,
    )

    plt.gca().invert_yaxis()
    plt.gca().set_aspect(
        "equal"
    )

    plt.xlabel(
        "Spatial x"
    )

    plt.ylabel(
        "Spatial y"
    )

    plt.title(
        "Spatial distribution of morphology PC1"
    )

    cbar = plt.colorbar(
        scatter
    )

    cbar.set_label(
        "Image PC1"
    )

    plt.tight_layout()

    plt.savefig(
        FIGDIR /
        "04_morphology_PC1_spatial.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =========================================================
# 12. Spatial visualization of morphology PC2
# =========================================================

if (
    embedding_pca.shape[1] >= 2
    and "image_x" in manifest.columns
    and "image_y" in manifest.columns
):

    coords_x = manifest[
        "image_x"
    ].values

    coords_y = manifest[
        "image_y"
    ].values

    plt.figure(
        figsize=(8, 8)
    )

    scatter = plt.scatter(
        coords_x,
        coords_y,
        c=embedding_pca[:, 1],
        s=10,
    )

    plt.gca().invert_yaxis()
    plt.gca().set_aspect(
        "equal"
    )

    plt.xlabel(
        "Spatial x"
    )

    plt.ylabel(
        "Spatial y"
    )

    plt.title(
        "Spatial distribution of morphology PC2"
    )

    cbar = plt.colorbar(
        scatter
    )

    cbar.set_label(
        "Image PC2"
    )

    plt.tight_layout()

    plt.savefig(
        FIGDIR /
        "05_morphology_PC2_spatial.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# =========================================================
# 13. Save summary
# =========================================================

summary = pd.DataFrame(
    {
        "metric": [
            "n_patches",
            "embedding_model",
            "embedding_dimension",
            "pca_dimension",
            "device",
            "embedding_nan_count",
            "pca_explained_variance_total",
        ],

        "value": [
            len(manifest),
            "ResNet50_ImageNet",
            embeddings.shape[1],
            embedding_pca.shape[1],
            str(DEVICE),
            int(np.isnan(embeddings).sum()),
            float(
                pca.explained_variance_ratio_.sum()
            ),
        ],
    }
)

summary.to_csv(
    OUTDIR /
    "image_embedding_summary.csv",
    index=False,
)


# =========================================================
# 14. Final report
# =========================================================

print("\nSaved:")

print(
    "outputs/resnet50_image_embeddings.npy"
)

print(
    "outputs/resnet50_image_embeddings_pca.npy"
)

print(
    "outputs/spot_image_embedding_manifest.csv"
)

print(
    "outputs/image_embedding_pca.csv"
)

print(
    "outputs/image_embedding_summary.csv"
)

if "cluster" in manifest.columns:
    print(
        "figures/03_resnet50_embedding_pca.png"
    )

if (
    "image_x" in manifest.columns
    and "image_y" in manifest.columns
):

    print(
        "figures/04_morphology_PC1_spatial.png"
    )

    print(
        "figures/05_morphology_PC2_spatial.png"
    )


print(
    "\nD9 image embedding extraction complete."
)
