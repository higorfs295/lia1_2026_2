# =====================================================================
# CÉLULA: coleta de predições + motor de laudo estruturado
# =====================================================================
import os, glob, json, math, datetime
import numpy as np
import cv2
import sys
from .avaliacao import consolidar, iou_matriz
from .sintetico import mascara_encefalo, carregar_cinza, yolo_para_pixel

VERSAO_LAUDO = "1.0"

# ------------------------------------------------------ coleta em lote
def coletar(modelo, caminhos_img, raiz_labels=None, conf_min=0.05, imgsz=640,
            iou_consolidacao=0.55, lote=16, device=None, augment=False):
    """
    Roda o modelo sobre uma lista de imagens e devolve `registros`
    no formato consumido por c_avaliacao (GT em pixels, predição em pixels).
    conf_min baixo de propósito: as curvas PR precisam da cauda de baixa confiança.
    """
    registros = []
    for i in range(0, len(caminhos_img), lote):
        bloco = caminhos_img[i:i + lote]
        preds = modelo.predict(bloco, imgsz=imgsz, conf=conf_min, verbose=False,
                               device=device, augment=augment)
        for cam, p in zip(bloco, preds):
            H, W = p.orig_shape
            b = p.boxes
            cx = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
            cf = b.conf.cpu().numpy() if len(b) else np.zeros(0)
            cl = b.cls.cpu().numpy().astype(int) if len(b) else np.zeros(0, int)
            cx, cf, cl = consolidar(cx, cf, cl, iou_thr=iou_consolidacao)
            gt_cx, gt_cls = [], []
            if raiz_labels:
                lp = os.path.join(raiz_labels,
                                  os.path.splitext(os.path.basename(cam))[0] + ".txt")
                if os.path.exists(lp):
                    for l in open(lp).read().strip().split("\n"):
                        if not l.strip():
                            continue
                        q = l.split()
                        gt_cls.append(int(q[0]))
                        gt_cx.append(yolo_para_pixel(*[float(v) for v in q[1:5]], W, H))
            registros.append(dict(caminho=cam, largura=W, altura=H,
                                  gt_cx=gt_cx, gt_cls=gt_cls,
                                  pr_cx=cx.tolist(), pr_cf=cf.tolist(),
                                  pr_cls=cl.tolist()))
    return registros

# --------------------------------------------------- métricas de imagem
def _linha_media(masc):
    """Estimativa da linha média sagital: centróide horizontal do encéfalo."""
    ys, xs = np.where(masc > 0)
    return float(xs.mean()) if len(xs) else masc.shape[1] / 2

def _descrever_lesao(img, masc, caixa, area_encefalo, x_media, idx, classe,
                     nome_classe, conf):
    x1, y1, x2, y2 = [float(v) for v in caixa]
    w, h = x2 - x1, y2 - y1
    area = w * h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    recorte = img[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]
    intensidade = float(recorte.mean()) if recorte.size else float("nan")
    dentro = img[masc > 0]
    ref = float(dentro.mean()) if dentro.size else float(img.mean())
    return {
        "id": f"L{idx+1:02d}",
        "achado": nome_classe,
        "classe_id": int(classe),
        "confianca": round(float(conf), 4),
        "caixa_px": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "centro_px": [round(cx, 1), round(cy, 1)],
        "largura_px": round(w, 1), "altura_px": round(h, 1),
        "area_px2": round(area, 1),
        "area_relativa_encefalo_pct": round(100 * area / area_encefalo, 3) if area_encefalo else None,
        "diametro_equivalente_px": round(2 * math.sqrt(area / math.pi), 1),
        "razao_aspecto": round(w / h, 3) if h else None,
        "lateralidade_imagem": "esquerda da imagem" if cx < x_media else "direita da imagem",
        "deslocamento_da_linha_media_px": round(cx - x_media, 1),
        "intensidade_media": round(intensidade, 1),
        "contraste_relativo": round(intensidade / ref, 3) if ref else None,
    }

def gerar_laudo(modelo, caminho_img, nomes_classes, limiar=0.25, imgsz=640,
                iou_consolidacao=0.55, device=None, consistencia=True,
                escala_mm_por_px=None):
    """
    Executa o pipeline completo numa única imagem e devolve um laudo estruturado.
    `escala_mm_por_px` é OPCIONAL: sem o cabeçalho DICOM não há como inferir a
    escala física, então as medidas saem em pixels por padrão.
    """
    img = carregar_cinza(caminho_img)
    H, W = img.shape
    masc = mascara_encefalo(img)
    area_encefalo = float((masc > 0).sum())
    x_media = _linha_media(masc)

    p = modelo.predict([caminho_img], imgsz=imgsz, conf=limiar, verbose=False,
                       device=device)[0]
    b = p.boxes
    cx = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
    cf = b.conf.cpu().numpy() if len(b) else np.zeros(0)
    cl = b.cls.cpu().numpy().astype(int) if len(b) else np.zeros(0, int)
    cx, cf, cl = consolidar(cx, cf, cl, iou_thr=iou_consolidacao)

    # ---- consistência sob transformações (proxy honesto de incerteza)
    #      A cabeça end-to-end do YOLO26 não aceita `augment=True`; então a
    #      estabilidade é medida repetindo a inferência com (a) espelhamento
    #      horizontal e (b) outra escala de entrada, e casando as caixas por IoU.
    estabilidade = None
    if consistencia and len(cx):
        vistas = []
        # (a) espelhamento horizontal — também sonda viés esquerda/direita
        espelhada = cv2.flip(img, 1)
        tmp = os.path.join("/tmp", "_flip_" + os.path.basename(caminho_img))
        cv2.imwrite(tmp, espelhada)
        pf = modelo.predict([tmp], imgsz=imgsz, conf=limiar, verbose=False, device=device)[0]
        bf = pf.boxes
        if len(bf):
            cf_box = bf.xyxy.cpu().numpy().copy()
            cf_box[:, [0, 2]] = W - cf_box[:, [2, 0]]       # desfaz o espelhamento
            vistas.append(cf_box)
        else:
            vistas.append(np.zeros((0, 4)))
        try:
            os.remove(tmp)
        except OSError:
            pass
        # (b) outra escala de entrada
        alt = int(imgsz * 0.75) // 32 * 32
        pe = modelo.predict([caminho_img], imgsz=max(alt, 320), conf=limiar,
                            verbose=False, device=device)[0]
        vistas.append(pe.boxes.xyxy.cpu().numpy() if len(pe.boxes) else np.zeros((0, 4)))
        acumulado = []
        for v in vistas:
            acumulado.append(iou_matriz(cx, v).max(axis=1) if len(v) else np.zeros(len(cx)))
        estabilidade = [round(float(v), 3) for v in np.mean(acumulado, axis=0)]

    lesoes = []
    for i in range(len(cx)):
        d = _descrever_lesao(img, masc, cx[i], area_encefalo, x_media, i,
                             cl[i], nomes_classes[int(cl[i])], cf[i])
        if estabilidade is not None:
            d["estabilidade_iou"] = estabilidade[i]
        if escala_mm_por_px:
            d["diametro_equivalente_mm"] = round(
                d["diametro_equivalente_px"] * escala_mm_por_px, 2)
            d["area_mm2"] = round(d["area_px2"] * escala_mm_por_px ** 2, 1)
        lesoes.append(d)
    lesoes.sort(key=lambda d: -d["confianca"])
    for i, d in enumerate(lesoes):
        d["id"] = f"L{i+1:02d}"

    carga = sum(d["area_px2"] for d in lesoes)
    laudo = {
        "versao_laudo": VERSAO_LAUDO,
        "gerado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "arquivo": os.path.basename(caminho_img),
        "imagem": {"largura_px": W, "altura_px": H,
                   "area_intracraniana_px2": int(area_encefalo),
                   "linha_media_estimada_px": round(x_media, 1)},
        "parametros": {"limiar_confianca": limiar, "imgsz": imgsz,
                       "iou_consolidacao": iou_consolidacao, "teste_consistencia": bool(consistencia),
                       "escala_mm_por_px": escala_mm_por_px},
        "resumo": {
            "n_achados": len(lesoes),
            "veredito": "achado detectável" if lesoes else "sem achado acima do limiar",
            "confianca_maxima": round(float(max([d["confianca"] for d in lesoes])), 4) if lesoes else 0.0,
            "carga_lesional_px2": round(carga, 1),
            "carga_lesional_pct_encefalo": round(100 * carga / area_encefalo, 3) if area_encefalo else None,
            "classes_detectadas": sorted({d["achado"] for d in lesoes}),
        },
        "achados": lesoes,
        "aviso": ("Saída de um exercício acadêmico de visão computacional. "
                  "NÃO é dispositivo médico, NÃO tem validação clínica e "
                  "NÃO deve ser usada para decisão diagnóstica."),
    }
    return laudo, p

def laudo_em_texto(laudo):
    r = laudo["resumo"]; L = []
    L.append(f"LAUDO AUTOMÁTICO (v{laudo['versao_laudo']}) — {laudo['arquivo']}")
    L.append(f"Gerado em {laudo['gerado_em']}")
    L.append("-" * 64)
    L.append(f"Veredito.................: {r['veredito']}")
    L.append(f"Achados acima do limiar..: {r['n_achados']}  "
             f"(limiar τ = {laudo['parametros']['limiar_confianca']})")
    L.append(f"Confiança máxima.........: {r['confianca_maxima']:.3f}")
    L.append(f"Carga lesional...........: {r['carga_lesional_px2']:.0f} px² "
             f"({r['carga_lesional_pct_encefalo']}% da área intracraniana)")
    L.append(f"Classes..................: {', '.join(r['classes_detectadas']) or '—'}")
    L.append("-" * 64)
    for d in laudo["achados"]:
        L.append(f"[{d['id']}] {d['achado']}  conf={d['confianca']:.3f}"
                 + (f"  estabilidade={d['estabilidade_iou']:.2f}" if "estabilidade_iou" in d else ""))
        L.append(f"      caixa {d['caixa_px']}  centro {d['centro_px']}")
        L.append(f"      área {d['area_px2']:.0f} px² ({d['area_relativa_encefalo_pct']}% do encéfalo)"
                 f"  Ø equiv. {d['diametro_equivalente_px']:.0f} px")
        L.append(f"      {d['lateralidade_imagem']} (Δ linha média {d['deslocamento_da_linha_media_px']:+.0f} px)"
                 f"  contraste rel. {d['contraste_relativo']}")
    L.append("-" * 64)
    L.append(laudo["aviso"])
    return "\n".join(L)
