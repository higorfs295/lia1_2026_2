# =====================================================================
# CÉLULA: benchmark de robustez sob artefatos de RM
# =====================================================================
import os, glob, shutil, sys
import cv2, numpy as np
from .sintetico import ARTEFATOS, aplicar_artefato, carregar_cinza
from .inferencia import coletar
from .avaliacao import mapa

def preparar_corrompido(imagens, raiz_labels, destino, artefato, severidade):
    di = os.path.join(destino, "images"); dl = os.path.join(destino, "labels")
    os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
    for ip in imagens:
        nome = os.path.basename(ip); stem = os.path.splitext(nome)[0]
        img = carregar_cinza(ip)
        cv2.imwrite(os.path.join(di, nome), aplicar_artefato(img, artefato, severidade))
        lp = os.path.join(raiz_labels, stem + ".txt")
        shutil.copy(lp, os.path.join(dl, stem + ".txt")) if os.path.exists(lp) \
            else open(os.path.join(dl, stem + ".txt"), "w").close()
    return sorted(glob.glob(di + "/*")), dl

def avaliar_robustez(modelo, imagens, raiz_labels, cfg, n_classes=2,
                     artefatos=None, severidades=(1, 2, 3), tmp="/tmp/robustez"):
    """
    Devolve {artefato: {0: mAP50_original, 1: ..., 2: ..., 3: ...}}.
    A severidade 0 é sempre a imagem intacta (mesma amostra, base comparável).
    """
    artefatos = artefatos or list(ARTEFATOS)
    base = coletar(modelo, imagens, raiz_labels=raiz_labels,
                   conf_min=cfg.conf_coleta, imgsz=cfg.imgsz,
                   iou_consolidacao=cfg.iou_consolidacao, device=cfg.device)
    m0 = mapa(base, n_classes, ious=[0.5])["map50"]
    tabela = {}
    for a in artefatos:
        tabela[a] = {0: m0}
        for s in severidades:
            d = os.path.join(tmp, f"{a}_{s}")
            shutil.rmtree(d, ignore_errors=True)
            imgs_c, lab_c = preparar_corrompido(imagens, raiz_labels, d, a, s)
            regs = coletar(modelo, imgs_c, raiz_labels=lab_c,
                           conf_min=cfg.conf_coleta, imgsz=cfg.imgsz,
                           iou_consolidacao=cfg.iou_consolidacao, device=cfg.device)
            tabela[a][s] = mapa(regs, n_classes, ious=[0.5])["map50"]
    return tabela, m0

def taxa_alarme_falso_ood(modelo, imagens_ood, limiar, cfg):
    """
    Controle negativo: imagens médicas de OUTRO domínio (medical-pills), onde
    nenhum achado é possível. Mede com que frequência o pipeline inventa lesão.
    """
    regs = coletar(modelo, imagens_ood, raiz_labels=None, conf_min=limiar,
                   imgsz=cfg.imgsz, iou_consolidacao=cfg.iou_consolidacao,
                   device=cfg.device)
    com_alarme = sum(1 for r in regs if len(r["pr_cf"]) > 0)
    total_caixas = sum(len(r["pr_cf"]) for r in regs)
    confs = [c for r in regs for c in r["pr_cf"]]
    return dict(imagens=len(regs), imagens_com_alarme=com_alarme,
                taxa_alarme_falso=com_alarme / max(len(regs), 1),
                caixas_espurias=total_caixas,
                caixas_por_imagem=total_caixas / max(len(regs), 1),
                confianca_media=float(np.mean(confs)) if confs else 0.0,
                confianca_maxima=float(np.max(confs)) if confs else 0.0)
