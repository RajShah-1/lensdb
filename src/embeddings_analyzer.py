import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  needed for 3D plot


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


if __name__ == "__main__":
    analyze_embeddings("demo")
