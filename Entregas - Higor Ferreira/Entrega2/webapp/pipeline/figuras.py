# =====================================================================
# CÉLULA: biblioteca de figuras do projeto (todas na paleta do padrão)
# =====================================================================
import os, math, collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
from .estilo import PALETA, CICLO, CMAP_DTOX, titular, moldura

COR_CLASSE = {0: PALETA["primaria"], 1: PALETA["roxo"]}
COR_STATUS = {"TP": PALETA["verde"], "FP": PALETA["magenta"], "FN": PALETA["laranja"]}

def salvar(fig, caminho):
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    moldura(fig); fig.savefig(caminho); return caminho

# ---------------------------------------------------------- 1. EDA
def fig_composicao_particoes(resumo, caminho=None):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    nomes = [r["particao"] for r in resumo]
    x = np.arange(len(nomes)); larg = 0.38
    ax[0].bar(x - larg/2, [r["negative"] for r in resumo], larg,
              label="negative", color=PALETA["primaria"])
    ax[0].bar(x + larg/2, [r["positive"] for r in resumo], larg,
              label="positive", color=PALETA["roxo"])
    ax[0].set_xticks(x); ax[0].set_xticklabels(nomes); ax[0].legend()
    titular(ax[0], "Caixas por classe", "desequilíbrio muda entre partições")
    ax[1].bar(x, [r["imagens"] for r in resumo], 0.55, color=PALETA["primaria"])
    for i, r in enumerate(resumo):
        ax[1].text(i, r["imagens"], str(r["imagens"]), ha="center", va="bottom", fontsize=9)
    ax[1].set_xticks(x); ax[1].set_xticklabels(nomes)
    titular(ax[1], "Imagens por partição")
    prop = [100 * r["positive"] / max(r["negative"] + r["positive"], 1) for r in resumo]
    ax[2].bar(x, prop, 0.55, color=[PALETA["primaria"], PALETA["primaria"], PALETA["magenta"]][:len(x)])
    ax[2].axhline(prop[0], ls="--", lw=1.4, color=PALETA["texto"])
    for i, v in enumerate(prop):
        ax[2].text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax[2].set_xticks(x); ax[2].set_xticklabels(nomes); ax[2].set_ylabel("% positive")
    titular(ax[2], "Prevalência da classe positive", "a linha tracejada é o treino")
    fig.tight_layout()
    return salvar(fig, caminho) if caminho else fig

def fig_geometria_e_area(registros_por_particao, caminho=None):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    # geometrias
    todas = collections.Counter()
    for regs in registros_por_particao.values():
        for r in regs:
            todas[(r["largura"], r["altura"])] += 1
    itens = todas.most_common(8)
    rot = [f"{w}×{h}" for (w, h), _ in itens]
    ax[0].barh(rot[::-1], [v for _, v in itens][::-1], color=PALETA["primaria"])
    titular(ax[0], "Protocolos de aquisição", "geometria da imagem como proxy de fonte")
    # área relativa das caixas
    for i, (nome, regs) in enumerate(registros_por_particao.items()):
        areas = [c[3] * c[4] * 100 for r in regs for c in r["caixas"]]
        if areas:
            ax[1].hist(areas, bins=np.logspace(-1, 1.2, 30), histtype="step", lw=2,
                       label=nome, color=CICLO[i])
    ax[1].set_xscale("log"); ax[1].set_xlabel("área da caixa (% da imagem)")
    ax[1].legend(); titular(ax[1], "Escala dos alvos", "mediana ~1,5% → objeto pequeno")
    # nº de caixas por imagem
    for i, (nome, regs) in enumerate(registros_por_particao.items()):
        c = collections.Counter(r["n_caixas"] for r in regs)
        xs = sorted(c); ax[2].plot(xs, [c[k] for k in xs], "o-", label=nome, color=CICLO[i])
    ax[2].set_xlabel("caixas por imagem"); ax[2].set_yscale("symlog"); ax[2].legend()
    titular(ax[2], "Densidade de anotação")
    fig.tight_layout()
    return salvar(fig, caminho) if caminho else fig

def fig_mapa_calor_centros(registros_por_particao, caminho=None, n=64):
    fig, ax = plt.subplots(1, len(registros_por_particao),
                           figsize=(4.4 * len(registros_por_particao), 4.2))
    ax = np.atleast_1d(ax)
    for i, (nome, regs) in enumerate(registros_por_particao.items()):
        H = np.zeros((n, n))
        for r in regs:
            for c in r["caixas"]:
                xi = min(n - 1, int(c[1] * n)); yi = min(n - 1, int(c[2] * n))
                H[yi, xi] += 1
        H = np.clip(H, 0, np.percentile(H[H > 0], 99) if (H > 0).any() else 1)
        im = ax[i].imshow(H, cmap=CMAP_DTOX, origin="upper")
        ax[i].set_xticks([]); ax[i].set_yticks([])
        ax[i].set_title(f"{nome}", color=PALETA["escuro"], fontweight="bold")
        fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.03)
    fig.suptitle("Onde as lesões aparecem no campo de visão",
                 color=PALETA["escuro"], fontweight="bold")
    fig.tight_layout()
    return salvar(fig, caminho) if caminho else fig

# ------------------------------------------------ 2. curvas de treino
def fig_curvas_treino(csv_resultados, caminho=None):
    import pandas as pd
    df = pd.read_csv(csv_resultados)
    df.columns = [c.strip() for c in df.columns]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    perdas = [c for c in df.columns if c.startswith("train/")]
    for i, c in enumerate(perdas):
        ax[0].plot(df["epoch"], df[c], label=c.split("/")[-1], color=CICLO[i])
    ax[0].set_xlabel("época"); ax[0].legend(fontsize=8)
    titular(ax[0], "Perdas de treino")
    vperdas = [c for c in df.columns if c.startswith("val/")]
    for i, c in enumerate(vperdas):
        ax[1].plot(df["epoch"], df[c], label=c.split("/")[-1], color=CICLO[i])
    ax[1].set_xlabel("época"); ax[1].legend(fontsize=8)
    titular(ax[1], "Perdas de validação", "divergência aqui = sobreajuste")
    for i, c in enumerate([c for c in df.columns if c.startswith("metrics/")]):
        ax[2].plot(df["epoch"], df[c], label=c.replace("metrics/", "").replace("(B)", ""),
                   color=CICLO[i])
    ax[2].set_xlabel("época"); ax[2].legend(fontsize=8); ax[2].set_ylim(0, 1)
    titular(ax[2], "Métricas por época")
    fig.tight_layout()
    return salvar(fig, caminho) if caminho else fig

# ------------------------------------------------ 3. PR / F1 / limiar
def fig_pr_e_limiar(curvas, varredura, ponto, nomes, caminho=None):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for c, dados in curvas.items():
        if len(dados["recall"]) > 1:
            ax[0].plot(dados["recall"], dados["precision"], color=COR_CLASSE.get(c, CICLO[c]),
                       label=f"{nomes[c]} · AP@50={dados['ap']:.3f}")
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precisão")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1.02); ax[0].legend(fontsize=8)
    titular(ax[0], "Curva precisão–recall", "área sob a curva = AP")
    t = [l["limiar"] for l in varredura]
    ax[1].plot(t, [l["precisao"] for l in varredura], label="precisão", color=PALETA["primaria"])
    ax[1].plot(t, [l["recall"] for l in varredura], label="recall", color=PALETA["ciano"])
    ax[1].plot(t, [l["f1"] for l in varredura], label="F1", color=PALETA["roxo"], lw=2.8)
    if ponto["f1"]:
        ax[1].axvline(ponto["f1"]["limiar"], ls="--", color=PALETA["escuro"], lw=1.3)
        ax[1].annotate(f"τ*={ponto['f1']['limiar']:.2f}",
                       (ponto["f1"]["limiar"], ponto["f1"]["f1"]),
                       textcoords="offset points", xytext=(6, 8), fontsize=9,
                       color=PALETA["escuro"], fontweight="bold")
    if ponto.get("triagem"):
        ax[1].axvline(ponto["triagem"]["limiar"], ls=":", color=PALETA["magenta"], lw=1.6)
        ax[1].annotate(f"τ_triagem={ponto['triagem']['limiar']:.2f}",
                       (ponto["triagem"]["limiar"], 0.05), textcoords="offset points",
                       xytext=(6, 0), fontsize=8.5, color=PALETA["magenta"])
    ax[1].set_xlabel("limiar de confiança τ"); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1.02)
    titular(ax[1], "Escolha do ponto de operação", "não existe τ universal: existe τ para um objetivo")
    ax[2].plot([l["FP"] for l in varredura], [l["TP"] for l in varredura],
               color=PALETA["primaria"])
    ax[2].set_xlabel("falsos positivos acumulados"); ax[2].set_ylabel("verdadeiros positivos")
    titular(ax[2], "Compromisso TP × FP", "o eixo que o radiologista sente")
    fig.tight_layout()
    return salvar(fig, caminho) if caminho else fig

# ------------------------------------------------ 4. matriz de confusão
def fig_matriz_confusao(M, rotulos, caminho=None, titulo="Matriz de confusão"):
    Mn = M / np.maximum(M.sum(axis=0, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(Mn, cmap=CMAP_DTOX, vmin=0, vmax=1)
    ax.set_xticks(range(len(rotulos))); ax.set_xticklabels(rotulos, rotation=20)
    ax.set_yticks(range(len(rotulos))); ax.set_yticklabels(rotulos)
    ax.set_xlabel("verdadeiro"); ax.set_ylabel("predito")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]}\n{Mn[i,j]*100:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=PALETA["branco"] if Mn[i, j] < 0.55 else PALETA["escuro"])
    ax.grid(False); ax.set_title(titulo, color=PALETA["escuro"], fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return salvar(fig, caminho) if caminho else fig

# ------------------------------------------------ 5. calibração
def fig_calibracao(linhas, valor_ece, caminho=None):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    c = [l["centro"] for l in linhas]
    acc = [np.nan if l["n"] == 0 else l["acuracia"] for l in linhas]
    ax[0].plot([0, 1], [0, 1], ls="--", color=PALETA["texto"], label="calibração perfeita")
    ax[0].bar(c, acc, width=0.09, color=PALETA["primaria"], edgecolor=PALETA["escuro"],
              linewidth=0.8, label="acurácia observada")
    ax[0].set_xlabel("confiança predita"); ax[0].set_ylabel("fração de acertos")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
    titular(ax[0], f"Diagrama de confiabilidade  ·  ECE = {valor_ece:.3f}",
            "acima da diagonal = subconfiante; abaixo = superconfiante")
    ax[1].bar(c, [l["n"] for l in linhas], width=0.09, color=PALETA["ciano"])
    ax[1].set_xlabel("confiança predita"); ax[1].set_ylabel("nº de detecções")
    titular(ax[1], "Massa por faixa de confiança")
    fig.tight_layout()
    return salvar(fig, caminho) if caminho else fig

# ------------------------------------------------ 6. taxonomia de erros
def fig_taxonomia(tax, caminho=None):
    rot = {"classificacao": "classe errada (IoU ok)",
           "localizacao": "caixa mal posicionada",
           "cls+loc": "classe e caixa erradas",
           "duplicata": "duplicata",
           "fundo": "invenção sobre o fundo",
           "nao_detectado": "lesão não detectada"}
    itens = [(rot[k], v) for k, v in tax["erros"].items()]
    itens.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(8.2, 4))
    cores = [PALETA["magenta"] if "não detectada" in n else PALETA["primaria"]
             for n, _ in itens]
    ax.barh([n for n, _ in itens], [v for _, v in itens], color=cores)
    for i, (_, v) in enumerate(itens):
        ax.text(v, i, f" {v}", va="center", fontsize=9, color=PALETA["escuro"])
    titular(ax, f"Anatomia do erro  ·  {tax['tp']} acertos",
            "onde o modelo falha importa mais do que quanto ele falha")
    fig.tight_layout()
    return salvar(fig, caminho) if caminho else fig

# ------------------------------------------------ 7. robustez
def fig_robustez(tabela, caminho=None):
    """tabela: {artefato: {severidade: map50}} com severidade 0 = original."""
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for i, (nome, serie) in enumerate(tabela.items()):
        sev = sorted(serie)
        ax.plot(sev, [serie[s] for s in sev], "o-", label=nome, color=CICLO[i])
    ax.set_xlabel("severidade do artefato"); ax.set_ylabel("mAP@50")
    ax.set_xticks(sorted({s for v in tabela.values() for s in v}))
    ax.legend(fontsize=8.5)
    titular(ax, "Degradação sob artefatos de ressonância",
            "severidade 0 = imagem original; queda acentuada = fragilidade")
    fig.tight_layout()
    return salvar(fig, caminho) if caminho else fig

# ------------------------------------------------ 8. comparativo de domínios
def fig_dominios(resultados, caminho=None):
    """resultados: lista de dict(nome, map50, map50_95, recall, precisao)."""
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    nomes = [r["nome"] for r in resultados]; x = np.arange(len(nomes)); w = 0.38
    ax[0].bar(x - w/2, [r["map50"] for r in resultados], w, label="mAP@50", color=PALETA["primaria"])
    ax[0].bar(x + w/2, [r["map50_95"] for r in resultados], w, label="mAP@50-95", color=PALETA["turquesa"])
    for i, r in enumerate(resultados):
        ax[0].text(i - w/2, r["map50"], f"{r['map50']:.3f}", ha="center", va="bottom", fontsize=8)
        ax[0].text(i + w/2, r["map50_95"], f"{r['map50_95']:.3f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(nomes, fontsize=9); ax[0].legend(fontsize=8)
    ax[0].set_ylim(0, 1)
    titular(ax[0], "Generalização entre domínios", "queda interno→externo = custo do deslocamento")
    ax[1].bar(x - w/2, [r.get("precisao", 0) for r in resultados], w, label="precisão", color=PALETA["primaria"])
    ax[1].bar(x + w/2, [r.get("recall", 0) for r in resultados], w, label="recall", color=PALETA["ciano"])
    ax[1].set_xticks(x); ax[1].set_xticklabels(nomes, fontsize=9); ax[1].legend(fontsize=8)
    ax[1].set_ylim(0, 1)
    titular(ax[1], "Precisão e recall no ponto de operação")
    fig.tight_layout()
    return salvar(fig, caminho) if caminho else fig

# ------------------------------------------------ 9. grade qualitativa
def fig_qualitativa(registros, nomes, limiar=0.25, n=6, caminho=None, semente=0):
    import cv2
    from .avaliacao import casar_imagem
    rng = np.random.default_rng(semente)
    alvo = [r for r in registros if r["gt_cx"]]
    idx = rng.choice(len(alvo), size=min(n, len(alvo)), replace=False)
    cols = 3; linhas = math.ceil(len(idx) / cols)
    fig, ax = plt.subplots(linhas, cols, figsize=(4.2 * cols, 4.2 * linhas))
    ax = np.atleast_1d(ax).ravel()
    for k, i in enumerate(idx):
        r = alvo[i]
        img = cv2.imread(r["caminho"], cv2.IMREAD_GRAYSCALE)
        ax[k].imshow(img, cmap="gray"); ax[k].axis("off")
        # recorta o enquadramento no encéfalo para não desperdiçar área com fundo
        try:
            from .sintetico import mascara_encefalo
            ys, xs = np.where(mascara_encefalo(img) > 0)
            if len(xs):
                mg = 0.06 * max(img.shape)
                ax[k].set_xlim(max(0, xs.min()-mg), min(img.shape[1], xs.max()+mg))
                ax[k].set_ylim(min(img.shape[0], ys.max()+mg), max(0, ys.min()-mg))
        except Exception:
            pass
        keep = [j for j, c in enumerate(r["pr_cf"]) if c >= limiar]
        res, falt = casar_imagem(r["gt_cx"], r["gt_cls"],
                                 [r["pr_cx"][j] for j in keep],
                                 [r["pr_cf"][j] for j in keep],
                                 [r["pr_cls"][j] for j in keep])
        for (j, tipo, cat, _, _) in res:
            x1, y1, x2, y2 = r["pr_cx"][keep[j]]
            cor = COR_STATUS["TP"] if tipo == "TP" else COR_STATUS["FP"]
            ax[k].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec=cor, lw=2))
            ax[k].text(x1, y1 - 3, f"{nomes[r['pr_cls'][keep[j]]]} {r['pr_cf'][keep[j]]:.2f}",
                       color=cor, fontsize=7.5, fontweight="bold")
        for j in falt:
            x1, y1, x2, y2 = r["gt_cx"][j]
            ax[k].add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                      ec=COR_STATUS["FN"], lw=2, ls="--"))
            ax[k].text(x1, y2 + 12, "não detectado", color=COR_STATUS["FN"],
                       fontsize=7.5, fontweight="bold")
        ax[k].set_title(os.path.basename(r["caminho"]), fontsize=8, color=PALETA["escuro"])
    for a in ax[len(idx):]:
        a.axis("off")
    fig.suptitle("Verde = acerto · Magenta = falso positivo · Laranja tracejado = lesão perdida",
                 color=PALETA["escuro"], fontweight="bold", fontsize=11)
    fig.tight_layout()
    return salvar(fig, caminho) if caminho else fig
