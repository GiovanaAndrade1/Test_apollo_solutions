import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from classification.knn import run_knn

METRICS = [
    ("auc", "AUC", "Mean AUC"),
    ("f1_macro", "F1-Macro", "Mean F1-Macro"),
    ("top_k_accuracy", "Top-k Accuracy", "Mean Top-k Accuracy"),
]


def build_summary(df_euclidean, df_cosine):
    rows = []
    for distance, df in (("Euclidean", df_euclidean), ("Cosine", df_cosine)):
        for col, name, _ in METRICS:
            best = df.loc[df[col].idxmax()]
            rows.append(
                {
                    "Distance Metric": distance,
                    "Performance Metric": name,
                    "Best k": int(best["k"]),
                    "Score": best[col],
                }
            )
    return pd.DataFrame(rows)


def plot_metric(df_euclidean, df_cosine, column, name, ylabel):
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(x=df_euclidean["k"], y=df_euclidean[column], marker="o", label="Euclidean")
    sns.lineplot(x=df_cosine["k"], y=df_cosine[column], marker="o", label="Cosine")
    plt.title(f"{name} vs. k for Euclidean and Cosine Distances")
    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.xticks(df_euclidean["k"])
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    return fig


def evaluate(X, y):
    """Roda o KNN (Euclidiana e Cosseno), monta o resumo e gera os gráficos."""
    df_euclidean = pd.DataFrame(run_knn(X, y, "euclidean"))
    df_cosine = pd.DataFrame(run_knn(X, y, "cosine"))

    summary = build_summary(df_euclidean, df_cosine)
    print("\n===== FINAL RESULTS =====")
    print(summary.to_string(index=False))

    figures = {
        f"{column}_vs_k": plot_metric(df_euclidean, df_cosine, column, name, ylabel)
        for column, name, ylabel in METRICS
    }
    return df_euclidean, df_cosine, summary, figures
