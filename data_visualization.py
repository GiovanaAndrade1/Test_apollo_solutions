import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def plot_syndrome_distribution(y):
    distribution = pd.Series(y).value_counts()

    fig = plt.figure(figsize=(12, 6))
    sns.barplot(
        x=distribution.index.astype(str),
        y=distribution.values,
        hue=distribution.index.astype(str),
        palette="viridis",
        legend=False,
    )
    plt.title("Distribution of Syndrome IDs")
    plt.xlabel("Syndrome ID")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig


def plot_tsne(X, y):
    print("\nApplying t-SNE for dimensionality reduction...")
    X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)
    print("t-SNE completed.")

    df_tsne = pd.DataFrame(X_tsne, columns=["component_1", "component_2"])
    df_tsne["syndrome_id"] = y

    fig = plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x="component_1",
        y="component_2",
        hue="syndrome_id",
        palette="tab10",
        data=df_tsne,
        legend="full",
        alpha=0.7,
    )
    plt.title("t-SNE of Image Embeddings, Colored by Syndrome ID")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle="--", alpha=0.6)
    return fig
