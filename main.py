import argparse
import os

import matplotlib.pyplot as plt

import data_processing
import data_visualization
import evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="KNN Syndrome Classification")
    parser.add_argument("--data-path", default=data_processing.DEFAULT_DATA_PATH,
                        help="caminho do arquivo .p com os embeddings")
    parser.add_argument("--output-dir", default="results",
                        help="pasta onde os gráficos e o resumo são salvos")
    parser.add_argument("--no-show", action="store_true",
                        help="não abre as janelas dos gráficos (apenas salva)")
    parser.add_argument("--skip-tsne", action="store_true",
                        help="pula o t-SNE")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    figures = {}

    # 1 - Dados
    X, y_original, _ = data_processing.load_and_process_data(args.data_path)
    data_processing.run_eda(y_original)
    y, _ = data_processing.encode_labels(y_original)

    # 2 - Visualização
    figures["syndrome_distribution"] = data_visualization.plot_syndrome_distribution(y_original)
    if not args.skip_tsne:
        figures["tsne"] = data_visualization.plot_tsne(X, y_original)

    # 3 e 4 - Classificação (KNN) e avaliação
    _, _, summary, eval_figures = evaluation.evaluate(X, y)
    figures.update(eval_figures)

    # Saídas
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    for name, fig in figures.items():
        fig.savefig(os.path.join(args.output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    print(f"\nResultados salvos em '{args.output_dir}/'.")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
