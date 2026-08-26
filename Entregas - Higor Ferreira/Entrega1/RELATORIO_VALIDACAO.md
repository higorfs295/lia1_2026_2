# ✅ Relatório de validação

## Avaliação geral: pronto para compartilhar

A entrega foi validada como um fluxo único, com execução **1.1 → 1.2 → 1.3**.

## Execução

| Notebook | Células | Resultado | Tempo de referência* |
|---|---:|---|---:|
| Entrega 1.1 — Titanic | 59 | ✅ sem erros | 17,6 s |
| Entrega 1.2 — Proveniência IoT | 44 | ✅ sem erros | 23,3 s |
| Entrega 1.3 — RT-IoT2022 | 69 | ✅ sem erros | 20,0 s |

\* Tempos observados no ambiente local de validação; o Colab e o primeiro cache Parquet podem variar.

## Correções funcionais aplicadas

1. O caminho do Colab foi centralizado na primeira célula, com montagem do Drive e campo editável `PASTA_PROJETO`.
2. A verificação de arquivos agora falha cedo e informa exatamente o que está faltando.
3. `titanic.csv` e `iris.csv` são baixados automaticamente se estiverem ausentes.
4. O notebook 1.3 passou a gravar seu JSON antes da síntese, removendo uma dependência circular.
5. Os JSONs finais ficam em `resultados/`, como os notebooks esperam.
6. O cache Parquet é refeito quando o CSV é mais recente, evitando leitura de dados obsoletos.
7. A verificação de integridade deixou de depender de `wc`, tornando-se compatível com Colab, Windows e macOS.

## Validação dos dados

| Dataset | Linhas | Colunas | MD5 abreviado | Conferência |
|---|---:|---:|---|---|
| Titanic | 891 | 15 | `56f29cc0b807` | ✅ |
| IoT Security & Energy | 4.000 | 24 | `cd09503e4bec` | ✅ |
| RT-IoT2022 | 123.117 | 85 | `87ec8763e33f` | ✅ |

As convenções registradas são idênticas nos três resultados:

```python
{"quantil": "linear", "ddof": 1, "outlier": "1.5*IQR", "corr": "pearson"}
```

## Revisão visual

- 13 gráficos inspecionados.
- Barras de magnitude partem de zero, exceto a distribuição de classes que usa escala logarítmica explicitamente rotulada.
- Títulos, eixos, unidades e legendas estão visíveis.
- Dois gráficos com escala `symlog` foram alterados para `log10(1 + x)`, preservando zeros sem exibir marcas negativas artificiais.
- Paleta azul-marinho/dourado aplicada de forma consistente.
- Capas, resumos, roteiros e tabelas foram renderizados em HTML para inspeção de layout.

## Limitações declaradas

- A execução foi reproduzida localmente com o mesmo fluxo de células. A chamada visual de montagem do Google Drive só é ativada dentro do Colab.
- O notebook Titanic usa o espelho `seaborn-data`, não o `train.csv` com o esquema original da competição Kaggle.
- O veredito “sintético” da Entrega 1.2 é uma conclusão metodológica baseada em múltiplos sinais; não é uma prova isolada de autoria ou processo de geração.

## Resultado da QA

**Pronto para compartilhar.** Não restam erros de execução conhecidos. As limitações acima estão visíveis no guia e nos notebooks.
