# =====================================================================
# CÉLULA: configuração, cronômetro de orçamento e wrapper de treino
# =====================================================================
import os, time, json, glob, shutil, sys
from dataclasses import dataclass, field, asdict
import numpy as np

# --------------------------------------------------------------- config
@dataclass
class Config:
    modelo: str = "yolo26l.pt"        # requisito do projeto: L no mínimo
    imgsz: int = 640
    lote: int = 16                    # "auto" também é aceito pelo Ultralytics
    orcamento_total_min: float = 30.0
    orcamento_treino_min: float = 15.0   # teto rígido passado ao treinador
    epocas_teto: int = 120            # só um teto; `time` decide o real
    sementes: int = 42
    n_sinteticas: int = 300
    usar_sinteticas: bool = True
    recall_minimo_triagem: float = 0.85
    iou_consolidacao: float = 0.55
    conf_coleta: float = 0.05
    artefatos_amostra: int = 90       # imagens usadas no teste de robustez
    severidades: tuple = (1, 2, 3)
    device: int | str = 0
    raiz: str = "/content/projeto"

    def dicionario(self):
        d = asdict(self); d["severidades"] = list(d["severidades"]); return d

MODOS = {
    # perfis prontos; o orçamento de treino é o que realmente controla o relógio
    "rapido":   dict(modelo="yolo26s.pt", imgsz=512, orcamento_treino_min=6.0,
                     n_sinteticas=150, artefatos_amostra=60),
    "padrao":   dict(modelo="yolo26l.pt", imgsz=640, orcamento_treino_min=15.0,
                     n_sinteticas=300, artefatos_amostra=90),
    "completo": dict(modelo="yolo26x.pt", imgsz=640, orcamento_treino_min=45.0,
                     n_sinteticas=600, artefatos_amostra=223),
    # perfil de verificação: roda o notebook inteiro em CPU só para provar que
    # nenhuma célula quebra. Não produz resultado com significado científico.
    "teste":    dict(modelo="yolo26n.pt", imgsz=320, orcamento_treino_min=1.0,
                     n_sinteticas=20, artefatos_amostra=8, lote=4,
                     epocas_teto=3, severidades=(1, 3)),
}

def configurar(modo="padrao", **ajustes):
    c = Config(**MODOS[modo]); [setattr(c, k, v) for k, v in ajustes.items()]
    return c

# ------------------------------------------------------------ cronômetro
class Cronometro:
    """Mede cada etapa e confronta o total com o orçamento declarado."""
    def __init__(self, orcamento_min=30.0):
        self.t0 = time.time(); self.orcamento = orcamento_min * 60
        self.etapas = []; self._ultimo = self.t0

    def marco(self, nome, silencioso=False):
        agora = time.time()
        dt = agora - self._ultimo; self._ultimo = agora
        self.etapas.append(dict(etapa=nome, segundos=dt,
                                acumulado=agora - self.t0))
        if not silencioso:
            restante = self.orcamento - (agora - self.t0)
            print(f"⏱  {nome:<38} {dt:6.1f}s   "
                  f"acumulado {(agora-self.t0)/60:5.1f} min   "
                  f"restante {restante/60:5.1f} min")

    def restante_min(self):
        return (self.orcamento - (time.time() - self.t0)) / 60

    def tabela(self):
        import pandas as pd
        df = pd.DataFrame(self.etapas)
        df["minutos"] = df["segundos"] / 60
        df["% do orçamento"] = 100 * df["segundos"] / self.orcamento
        return df[["etapa", "minutos", "% do orçamento"]].round(2)

    def figura(self, caminho=None):
        import matplotlib.pyplot as plt
        from .estilo import PALETA, CICLO, titular, moldura
        fig, ax = plt.subplots(figsize=(9.5, max(3, 0.42 * len(self.etapas))))
        nomes = [e["etapa"] for e in self.etapas]
        mins = [e["segundos"] / 60 for e in self.etapas]
        inicio = np.concatenate([[0], np.cumsum(mins)[:-1]])
        cores = [PALETA["roxo"] if "reino" in n else PALETA["primaria"] for n in nomes]
        ax.barh(nomes[::-1], mins[::-1], left=inicio[::-1], color=cores[::-1])
        ax.axvline(self.orcamento / 60, color=PALETA["magenta"], ls="--", lw=1.8)
        ax.text(self.orcamento / 60, -0.6, f" orçamento {self.orcamento/60:.0f} min",
                color=PALETA["magenta"], fontsize=9, fontweight="bold")
        ax.set_xlabel("minutos desde o início")
        titular(ax, f"Onde foram os {sum(mins):.1f} minutos",
                "barra âmbar = treino; a linha vermelha é o teto declarado")
        fig.tight_layout(); moldura(fig)
        if caminho:
            os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
            fig.savefig(caminho); return caminho
        return fig

# ------------------------------------------------------------- treino
def montar_yaml(raiz, nome, treino_rel, val_rel, nomes):
    p = os.path.join(raiz, f"data_{nome}.yaml")
    bloco = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(nomes))
    open(p, "w").write(f"path: {os.path.abspath(raiz)}\ntrain: {treino_rel}\n"
                       f"val: {val_rel}\n\n{bloco}")
    return p

def treinar(cfg, yaml_treino, projeto, nome_run, cronometro=None):
    """
    O argumento `time` do Ultralytics é o que garante o orçamento: ele
    sobrescreve `epochs` e encerra o treino ao atingir o tempo declarado,
    devolvendo o melhor checkpoint até ali. Sem ele, o tempo total do
    notebook dependeria do hardware sorteado pelo Colab.
    """
    from ultralytics import YOLO
    modelo = YOLO(cfg.modelo)
    modelo.train(
        data=yaml_treino, epochs=cfg.epocas_teto, time=cfg.orcamento_treino_min / 60,
        imgsz=cfg.imgsz, batch=cfg.lote, device=cfg.device, seed=cfg.sementes,
        deterministic=False, cache="ram", workers=2, patience=100,
        cos_lr=True, close_mosaic=5, plots=True, val=True,
        project=projeto, name=nome_run, exist_ok=True, verbose=True,
    )
    pasta = os.path.join(projeto, nome_run)
    if cronometro:
        cronometro.marco("treino YOLO26")
    return YOLO(os.path.join(pasta, "weights", "best.pt")), pasta
