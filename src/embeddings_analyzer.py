import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def analyze_embeddings(video_name: str, n_components: int = 50, annotate: bool = True):
    emb_path = Path("data") / video_name / "embds.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")

    embs = np.load(emb_path)
    n_frames = embs.shape[0]
    print(f"Loaded embeddings {embs.shape}")

    frame_numbers = np.arange(n_frames)

    # Run PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embs)

    # Explained variance
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Variance Explained")
    plt.grid(True)
    plt.savefig(emb_path.parent / "pca_variance.png")
    plt.close()

    # 2D scatter (first 2 PCs)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=20)
    if annotate:
        for i, frame_no in enumerate(frame_numbers):
            if i % 20 == 0:  # annotate every 20th frame to avoid clutter
                plt.annotate(str(frame_no), (reduced[i, 0], reduced[i, 1]), fontsize=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection (first 2 components)")
    plt.savefig(emb_path.parent / "pca_scatter.png")
    plt.close()

    # 3D scatter (first 3 PCs)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], alpha=0.6, s=20)
    if annotate:
        for i, frame_no in enumerate(frame_numbers):
            if i % 20 == 0:  # annotate every 20th frame
                ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], str(frame_no), fontsize=6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA Projection (first 3 components)")
    plt.savefig(emb_path.parent / "pca_scatter_3d.png")
    plt.close()

    print(f"Saved PCA plots under {emb_path.parent}")

def analyze_embeddings_with_detection(video_name: str,
                                      n_components: int = 50,
                                      annotate: bool = True):
    base_dir = Path("data") / video_name
    emb_path = base_dir / "embds.npy"
    counts_path = base_dir / "counts.csv"

    if not emb_path.exists() or not counts_path.exists():
        raise FileNotFoundError(f"Missing files: {emb_path}, {counts_path}")

    # Load embeddings + detection counts
    embs = np.load(emb_path)
    counts = pd.read_csv(counts_path)

    if len(embs) != len(counts):
        raise ValueError(
            f"Mismatch: {len(embs)} embeddings vs {len(counts)} detection rows"
        )

    print(f"Loaded {embs.shape[0]} embeddings with detection counts")

    # Run PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embs)

    # Explained variance
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Variance Explained")
    plt.grid(True)
    plt.savefig(base_dir / "pca_variance_with_detection.png")
    plt.close()

    # 2D scatter, color by car count
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                          c=counts["car_count"], cmap="Reds", alpha=0.7)
    plt.colorbar(scatter, label="Car count")
    if annotate:
        for i, row in counts.iterrows():
            if i % 20 == 0:  # annotate every 20th
                plt.annotate(str(row["frame_id"]),
                             (reduced[i, 0], reduced[i, 1]),
                             fontsize=7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (colored by car count)")
    plt.savefig(base_dir / "pca_scatter_car.png")
    plt.close()

    # 2D scatter, color by people count
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                          c=counts["people_count"], cmap="Blues", alpha=0.7)
    plt.colorbar(scatter, label="People count")
    if annotate:
        for i, row in counts.iterrows():
            if i % 20 == 0:
                plt.annotate(str(row["frame_id"]),
                             (reduced[i, 0], reduced[i, 1]),
                             fontsize=7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (colored by people count)")
    plt.savefig(base_dir / "pca_scatter_people.png")
    plt.close()

    # 3D scatter (first 3 PCs), colored by car count
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                   c=counts["car_count"], cmap="Reds", alpha=0.7)
    fig.colorbar(p, label="Car count")
    if annotate:
        for i, row in counts.iterrows():
            if i % 20 == 0:
                ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2],
                        str(row["frame_id"]), fontsize=6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA (colored by car count)")
    plt.savefig(base_dir / "pca_scatter_3d_car.png")
    plt.close()

    print(f"Saved plots in {base_dir}")


if __name__ == "__main__":
    analyze_embeddings_with_detection("demo")

