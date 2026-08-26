"""relatorio.py — gabarito de relatório técnico para a Entrega 1.

Estrutura: identificação → escopo → rastreabilidade → convenções → auditoria →
análise → conclusões → apêndices. Cada análise segue o ciclo
Objetivo / Método / Resultado / Interpretação.

Registro formal, sem ornamentação. Toda escolha metodológica é declarada no
ponto em que é feita e consolidada no Apêndice A.
"""
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

SETUP = '''# Preparação do ambiente de execução.
import os, sys, json, math

DADOS = "."                      # Colab: "/content/drive/MyDrive/Entrega1"
sys.path.insert(0, DADOS); os.chdir(DADOS)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import nucleo as nu               # convenções, carga, auditoria, paleta

nu.aplicar_estilo()
pd.set_option("display.width", 130)
os.makedirs("resultados", exist_ok=True)

print("Convenções estatísticas fixadas:")
for k, v in nu.CONVENCOES.items():
    print(f"   {k:<10} {v}")'''

CALIBRACAO = '''# Regressão de convenções. Verifica 6 grupos do Titanic contra valores
# conhecidos. Interrompe a execução se qualquer convenção tiver mudado.
assert nu.calibrar("titanic.csv"), "convenções divergiram — investigar antes de prosseguir"'''


def capa(numero, titulo, resumo, identificacao):
    ident = "\n".join(f"| {k} | {v} |" for k, v in identificacao)
    return new_markdown_cell(f"""# Entrega {numero} — {titulo}

{resumo}

| Campo | Valor |
|---|---|
{ident}
""")


def escopo(questao, papel, prereq):
    p = "\n".join(f"- {x}" for x in prereq)
    return new_markdown_cell(f"""---

## 1. Escopo

### 1.1 Questão de pesquisa

> {questao}

### 1.2 Papel na Entrega 1

{papel}

### 1.3 Pré-requisitos de execução

{p}
""")


def rastreabilidade(linhas):
    corpo = "\n".join(f"| {r} | {s} | {e} |" for r, s, e in linhas)
    return new_markdown_cell(f"""---

## 2. Matriz de rastreabilidade

Mapeamento entre os requisitos do Desafio Final e as seções que os atendem.
A terceira coluna registra a evidência produzida, para que a verificação não
dependa de leitura integral.

| Requisito | Seção | Evidência produzida |
|---|---|---|
{corpo}
""")


def convencoes(premissas_lista):
    corpo = "\n".join(f"**P{i}. {t}** — {d}\n" for i, (t, d) in enumerate(premissas_lista, 1))
    return new_markdown_cell(f"""---

## 3. Convenções e premissas

### 3.1 Convenções estatísticas

Fixadas em `nucleo.CONVENCOES` e idênticas nas três entregas. A célula de
calibração da Seção 4 interrompe a execução se qualquer uma divergir.

| Convenção | Valor adotado | Alternativa comum | Efeito da troca |
|---|---|---|---|
| Quantil | interpolação linear (tipo 7 de Hyndman–Fan) | tipo 1 (sem interpolação) | altera contagem de outliers |
| Desvio padrão | `ddof=1` (amostral) | `ddof=0` (populacional, padrão do NumPy) | altera dispersão reportada |
| Outlier | regra de Tukey, `1,5 × IQR` | `3σ` | altera critério de exclusão |
| Correlação | Pearson | Spearman | altera sensibilidade a não-linearidade |

### 3.2 Premissas declaradas

{corpo}
""")


def secao(num, titulo, texto=""):
    return new_markdown_cell(f"---\n\n## {num}. {titulo}\n\n{texto}".rstrip())


def analise(num, titulo, objetivo, metodo, formalizacao=None):
    f = f"\n\n**Formalização.**\n\n{formalizacao}" if formalizacao else ""
    return new_markdown_cell(
        f"""### {num} {titulo}

**Objetivo.** {objetivo}

**Método.** {metodo}{f}""")


def resultado(texto):
    return new_markdown_cell(f"**Resultado e interpretação.**\n\n{texto}")


def apendice_decisoes(linhas):
    corpo = "\n".join(f"| {d} | {o} | {j} |" for d, o, j in linhas)
    return new_markdown_cell(f"""---

## Apêndice A — Registro de decisões

Decisões metodológicas tomadas nesta entrega, com a alternativa descartada e a
justificativa. Serve para que a revisão questione a escolha, não a omissão.

| Decisão | Alternativa descartada | Justificativa |
|---|---|---|
{corpo}
""")


def apendice_api(blocos):
    partes = [f"**{n}**\n\n```python\n{c}\n```" for n, c in blocos]
    return new_markdown_cell("---\n\n## Apêndice B — Referência rápida\n\n" +
                             "\n\n".join(partes))


def salvar(nb, arquivo, nome_colab):
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": [], "name": nome_colab, "toc_visible": True},
    }
    import nbformat as nbf
    nbf.write(nb, arquivo)
    print("gerado:", arquivo)
