# KNN Syndrome Classification

Estrutura:
- `main.py`: ponto de entrada único, roda todo o fluxo.
- `data_processing.py`: carrega, achata e prepara X/y.
- `data_visualization.py`: distribuição das classes e t-SNE.
- `classification/knn.py`: KNN com distância Euclidiana e Cosseno (validação cruzada estratificada).
- `classification/metrics.py`: AUC, F1-Macro e Top-k Accuracy.
- `evaluation.py`: melhores k, tabela-resumo e gráficos de avaliação.

Para rodar tudo:
```
pip install -r requirements.txt
python main.py --data-path mini_gm_public_v0.1.p
```

Opções: `--output-dir results`, `--no-show` (só salva os gráficos), `--skip-tsne`.

## Dados

O arquivo `mini_gm_public_v0.1.p` deve ser obtido separadamente e colocado na raiz do projeto, ou informado com `--data-path`. O conjunto de dados não é distribuído neste repositório.
