"""nucleo.py -- modulo compartilhado pelas 3 entregas.

Existe por um motivo: garantir que D1, D2 e D3 usem AS MESMAS convencoes.
Convencao divergente entre entregas e o erro que ninguem percebe -- foi o
que medimos: trocar a definicao de quantil muda a contagem de outliers da
setosa de 4 para 1, sem erro nenhum.

Nada aqui e otimizacao prematura: o cache em parquet foi medido em 10x
(84 colunas) a 106x (recorte de colunas) contra read_csv.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- convencoes
CONVENCOES = {
    "quantil": "linear",   # == tipo 7 do R == padrao numpy/pandas. FIXADO.
    "ddof": 1,             # amostral. pandas usa 1, numpy usa 0: declaramos.
    "outlier": "1.5*IQR",
    "corr": "pearson",
}


def quantis(s: pd.Series, qs=(0.25, 0.50, 0.75)) -> pd.Series:
    """Uma chamada, nao tres: medido 2x mais rapido."""
    return s.quantile(list(qs), interpolation=CONVENCOES["quantil"])


def desvio(s: pd.Series) -> float:
    return s.std(ddof=CONVENCOES["ddof"])


def outliers(s: pd.Series) -> pd.Series:
    """Mascara booleana pela regra 1.5*IQR. Retorna mascara, nao contagem:
    quem chama decide se conta, filtra ou marca."""
    q1, q3 = s.quantile([0.25, 0.75], interpolation=CONVENCOES["quantil"])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def correlacoes(df: pd.DataFrame, por: str | None = None) -> pd.DataFrame:
    """Correlacao numerica. Se `por` for dado, calcula DENTRO de cada grupo.

    O parametro existe por causa do paradoxo de Simpson: no Iris, pooled da
    -0.118 para sepal_length x sepal_width, e por especie da +0.74/+0.53/+0.46.
    Correlacao agregada sobre grupos heterogeneos e armadilha, nao resultado.
    """
    num = df.select_dtypes(include=[np.number])
    if por is None:
        return num.corr(method=CONVENCOES["corr"])
    return df.groupby(por, observed=True)[num.columns].corr(method=CONVENCOES["corr"])


# ------------------------------------------------------------------ carga
def carregar(csv: str, colunas=None, categoricas=(), cache=True) -> pd.DataFrame:
    """Le com cache em parquet e o refaz quando o CSV for atualizado."""
    pq = os.path.splitext(csv)[0] + ".parquet"
    cache_atual = (
        cache
        and os.path.exists(pq)
        and os.path.getmtime(pq) >= os.path.getmtime(csv)
    )
    if cache and not cache_atual:
        pd.read_csv(csv).to_parquet(pq, index=False)
    df = (pd.read_parquet(pq, columns=colunas) if cache and os.path.exists(pq)
          else pd.read_csv(csv, usecols=colunas))
    for c in categoricas:
        if c in df.columns:
            df[c] = df[c].astype("category")   # medido: 7x menos RAM, groupby 1.8x
    return df


def integridade(csv: str, df: pd.DataFrame) -> dict:
    """Confere parser e arquivo bruto de forma portavel (Colab/Windows/macOS)."""
    with open(csv, "rb") as arquivo:
        linhas_brutas = sum(1 for _ in arquivo)
    cru = max(linhas_brutas - 1, 0)

    resumo = hashlib.md5()
    with open(csv, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    md5 = resumo.hexdigest()[:12]
    return {"md5": md5, "linhas_wc": cru, "linhas_parser": len(df),
            "confere": cru is None or cru == len(df)}


# ------------------------------------------------------------------ perfil
def perfil(df: pd.DataFrame) -> pd.DataFrame:
    """Auditoria por coluna. Separa nulo real de vazio semantico ('' nao e NaN
    -- foi o que deixou passar o nome vazio no notebook original)."""
    linhas = []
    for c in df.columns:
        s = df[c]
        vazios = (s.astype(str).str.strip() == "").sum() if s.dtype == object else 0
        linhas.append({
            "coluna": c, "dtype": str(s.dtype),
            "nulos": int(s.isna().sum()),
            "nulos_%": round(100 * s.isna().mean(), 2),
            "vazios_str": int(vazios),
            "unicos": int(s.nunique(dropna=True)),
            "constante": s.nunique(dropna=True) <= 1,
        })
    return pd.DataFrame(linhas).sort_values("nulos_%", ascending=False)


def suspeitos_de_vazamento(df: pd.DataFrame, alvo: str, limiar=0.99) -> list[str]:
    """Colunas que determinam o alvo quase perfeitamente. Em IDS, acuracia de
    99% quase sempre e vazamento -- 'alive' vs 'survived' no Titanic e o caso
    didatico. Isto SINALIZA para inspecao; nao decide nada sozinho."""
    achados = []
    for c in df.columns:
        if c == alvo:
            continue
        if df[c].nunique(dropna=True) > 1000:
            continue
        pureza = df.groupby(c, observed=True)[alvo].agg(
            lambda g: g.value_counts(normalize=True).max())
        if pureza.min() >= limiar:
            achados.append(c)
    return achados


# ------------------------------------------------------------------ calibracao
OURO_TITANIC = {
    ("female", "First"): (94, 82.664550), ("female", "Second"): (76, 22.0),
    ("female", "Third"): (144, 12.475),   ("male", "First"):   (122, 41.2625),
    ("male", "Second"): (108, 13.0),      ("male", "Third"):   (347, 7.925),
}


def calibrar(csv="titanic.csv") -> bool:
    """O Titanic e o teste de regressao das convencoes, nao uma entrega.
    Roda em ~0.1s. Se falhar, nao confie em nada de D2 nem D3."""
    df = carregar(csv, colunas=["survived", "sex", "class", "age", "fare"],
                  categoricas=("sex", "class"))
    g = df.groupby(["sex", "class"], observed=True).agg(
        n=("survived", "size"), med=("fare", "median"))
    falhas = [k for k, (n, m) in OURO_TITANIC.items()
              if g.loc[k, "n"] != n or abs(g.loc[k, "med"] - m) > 1e-6]
    nulos_age = int(df["age"].isna().sum())
    if nulos_age != 177:
        falhas.append("age_nulos")
    print(f"calibracao: {'OK -- 6/6 grupos, age nulos=177' if not falhas else f'FALHOU {falhas}'}"
          f" | convencoes={CONVENCOES}")
    return not falhas


# ------------------------------------------------------------------ triagem
def _eta2(df: pd.DataFrame, num: str, alvo: str) -> float:
    """Eta ao quadrado: fracao da variancia de `num` explicada por `alvo`.
    0 = classe nao separa nada; 1 = separa perfeitamente (suspeite)."""
    s, g = df[num], df[alvo]
    ok = s.notna() & g.notna()
    s, g = s[ok], g[ok]
    total = ((s - s.mean()) ** 2).sum()
    if total == 0:
        return np.nan
    entre = df.loc[ok].groupby(g, observed=True)[num].agg(["size", "mean"])
    entre = (entre["size"] * (entre["mean"] - s.mean()) ** 2).sum()
    return float(entre / total)


def triagem(df: pd.DataFrame, alvo: str, top=12) -> dict:
    """Primeiro contato. NAO responde nada -- produz os fatos que decidem
    qual pergunta e respondivel neste dataset."""
    print(f"=== TRIAGEM: {df.shape[0]:,} linhas x {df.shape[1]} colunas | alvo='{alvo}' ===\n")

    vc = df[alvo].value_counts()
    razao = vc.max() / vc.min()
    print(f"[1] CLASSES ({len(vc)}) -- desbalanceamento {razao:.0f}:1")
    for k, v in vc.items():
        print(f"    {str(k)[:34]:<34}{v:>9,}  {100*v/len(df):5.2f}%")
    print(f"    acuracia da classe majoritaria sozinha: {100*vc.max()/len(df):.2f}%\n")

    mortas = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    print(f"[2] COLUNAS CONSTANTES (inuteis): {len(mortas)}"
          f"{' -> ' + ', '.join(mortas[:8]) if mortas else ''}\n")

    vaz = suspeitos_de_vazamento(df, alvo)
    print(f"[3] SUSPEITOS DE VAZAMENTO: {vaz if vaz else 'nenhum'}\n")

    num = [c for c in df.select_dtypes(include=[np.number]).columns if c != alvo]
    sep = pd.Series({c: _eta2(df, c, alvo) for c in num}).dropna().sort_values(ascending=False)
    print(f"[4] PODER DE SEPARACAO (eta^2), top {top} de {len(sep)} numericas:")
    for c, v in sep.head(top).items():
        marca = "  <-- separa quase perfeito, INVESTIGAR" if v > 0.95 else ""
        print(f"    {c:<28}{v:6.3f}{marca}")
    print(f"    features com eta^2 < 0.01 (ruido): {(sep < 0.01).sum()} de {len(sep)}\n")

    if len(sep) > 1:
        cc = df[sep.head(30).index].corr().abs()
        pares = [(a, b, cc.loc[a, b]) for i, a in enumerate(cc.columns)
                 for b in cc.columns[i+1:] if cc.loc[a, b] > 0.95]
        print(f"[5] PARES REDUNDANTES (|r|>0.95) entre as 30 mais separadoras: {len(pares)}")
        for a, b, r in pares[:6]:
            print(f"    {a} ~ {b}: {r:.3f}")
    print()
    return {"classes": vc, "desbalanceamento": razao, "constantes": mortas,
            "vazamento": vaz, "separacao": sep}


# ------------------------------------------------------ deteccao de sintetico
def detectar_sintetico(df: pd.DataFrame, alvo: str | None = None) -> dict:
    """Cinco testes de proveniencia. Nenhum e prova isolada; juntos, sim.

    Dado real de rede/energia tem cauda, correlacao entre grandezas fisicas e
    limites irregulares. Dado gerado por random.uniform(a, b) nao tem nada disso.
    """
    from scipy import stats
    num = [c for c in df.select_dtypes(include=[np.number]).columns if c != alvo]
    res, sinais = {}, 0

    unif = []
    for c in num:
        s = df[c].dropna()
        if s.max() > s.min() and s.nunique() > 20:
            if stats.kstest((s - s.min()) / (s.max() - s.min()), "uniform").pvalue > 0.05:
                unif.append(c)
    res["uniformes"] = (len(unif), len(num))
    sinais += len(unif) > len(num) / 2

    sem_out = [c for c in num if nu_outliers_zero(df[c])]
    res["sem_outliers"] = (len(sem_out), len(num))
    sinais += len(sem_out) > 0.9 * len(num)

    num_var = [c for c in num if df[c].nunique(dropna=True) > 1]   # constante envenena corr
    if len(num_var) > 1:
        cc = df[num_var].corr().abs().values
        rmax = float(cc[~np.eye(len(num_var), dtype=bool)].max())
        res["corr_max_entre_features"] = rmax
        sinais += rmax < 0.3

    if alvo is not None:
        eta = pd.Series({c: _eta2(df, c, alvo) for c in num}).dropna()
        res["eta2_max"] = float(eta.max()) if len(eta) else np.nan
        res["features_informativas"] = int((eta > 0.01).sum())
        sinais += res["features_informativas"] == 0

    res["sinais_de_sintetico"] = sinais
    res["veredito"] = ["real ou inconclusivo", "suspeito", "provavelmente sintetico",
                       "sintetico", "sintetico (inequivoco)"][min(sinais, 4)]
    return res


def nu_outliers_zero(s: pd.Series) -> bool:
    return s.dtype.kind in "if" and int(outliers(s).sum()) == 0


# ------------------------------------------------------------------ estilo
# Paleta IEEE UFG Student Branch: navy + gold, com meios-tons steel-blue.
PALETA = {
    "navy":  "#01172F", "gold":  "#D9A441", "steel": "#2E5C8A",
    "slate": "#5B7C99", "sand":  "#E8C87E", "rust":  "#A6522C",
    "cinza": "#8C8C8C",
}
SEQ = [PALETA["navy"], PALETA["gold"], PALETA["steel"],
       PALETA["rust"], PALETA["slate"], PALETA["sand"]]


def aplicar_estilo():
    """Estilo visual unico para as 3 entregas."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette=SEQ)
    except ImportError:
        pass
    mpl.rcParams.update({
        "figure.figsize": (10, 5), "figure.dpi": 110,
        "figure.facecolor": "white", "axes.facecolor": "#F8FAFC",
        "axes.prop_cycle": mpl.cycler(color=SEQ),
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlecolor": PALETA["navy"], "axes.labelsize": 10,
        "axes.edgecolor": "#CBD5E1", "grid.alpha": 0.22,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "legend.frameon": True, "legend.framealpha": .9,
    })
    return plt
