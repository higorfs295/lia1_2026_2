# =====================================================================
# CÉLULA: motor de avaliação — casamento, métricas, calibração, erros
# =====================================================================
import os, glob, math, json
import numpy as np

# --------------------------------------------------------------- IoU
def iou_matriz(a, b):
    """a:(N,4) b:(M,4) em xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    ab = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)

# --------------------------------------------- consolidação de detecções
def consolidar(caixas, confs, classes, iou_thr=0.55, agnostico=True):
    """
    Supressão gulosa de detecções redundantes.
    A cabeça do YOLO26 é end-to-end (sem NMS); ainda assim, modelos pouco
    treinados podem emitir duplicatas. Esta etapa deixa o laudo determinístico.
    """
    if len(caixas) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    caixas = np.asarray(caixas, np.float32); confs = np.asarray(confs, np.float32)
    classes = np.asarray(classes, int)
    ordem = np.argsort(-confs)
    mantidas = []
    for i in ordem:
        ok = True
        for j in mantidas:
            if not agnostico and classes[i] != classes[j]:
                continue
            if iou_matriz(caixas[i:i+1], caixas[j:j+1])[0, 0] > iou_thr:
                ok = False; break
        if ok:
            mantidas.append(i)
    m = np.array(mantidas, int)
    return caixas[m], confs[m], classes[m]

# ------------------------------------------------------------ casamento
CATEGORIAS_ERRO = ["classificacao", "localizacao", "cls+loc", "duplicata", "fundo"]

def casar_imagem(gt_cx, gt_cls, pr_cx, pr_cf, pr_cls, iou_tp=0.5, iou_loc=0.1):
    """
    Casamento guloso por confiança decrescente.
    Devolve, por predição, o rótulo TP/erro e, por GT, se foi encontrado.
    Taxonomia inspirada no TIDE (Bolya et al., 2020).
    """
    ordem = np.argsort(-np.asarray(pr_cf)) if len(pr_cf) else np.array([], int)
    usados = set(); resultado = []
    M = iou_matriz(pr_cx, gt_cx) if len(pr_cx) and len(gt_cx) else np.zeros((len(pr_cx), len(gt_cx)))
    for i in ordem:
        linha = M[i] if M.size else np.zeros(0)
        if linha.size == 0:
            resultado.append((i, "FP", "fundo", -1, 0.0)); continue
        j = int(np.argmax(linha)); v = float(linha[j])
        mesma = (gt_cls[j] == pr_cls[i])
        if v >= iou_tp and mesma and j not in usados:
            usados.add(j); resultado.append((i, "TP", None, j, v))
        elif v >= iou_tp and mesma and j in usados:
            resultado.append((i, "FP", "duplicata", j, v))
        elif v >= iou_tp and not mesma:
            resultado.append((i, "FP", "classificacao", j, v))
        elif iou_loc <= v < iou_tp and mesma:
            resultado.append((i, "FP", "localizacao", j, v))
        elif iou_loc <= v < iou_tp and not mesma:
            resultado.append((i, "FP", "cls+loc", j, v))
        else:
            resultado.append((i, "FP", "fundo", j, v))
    faltantes = [j for j in range(len(gt_cx)) if j not in usados]
    return resultado, faltantes

# ---------------------------------------------------- AP / curvas PR
def _ap_todos_pontos(rec, prec):
    """AP por interpolação em todos os pontos (convenção COCO/VOC2010+)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), mrec, mpre

def curvas_por_classe(registros, n_classes, iou_tp=0.5):
    """
    registros: lista por imagem de dict(gt_cx, gt_cls, pr_cx, pr_cf, pr_cls)
    Devolve, por classe: recall, precision, AP, e vetores conf/TP para calibração.
    """
    saida = {}
    for c in range(n_classes):
        confs, tps, n_gt = [], [], 0
        for r in registros:
            gt_m = [k for k, v in enumerate(r["gt_cls"]) if v == c]
            n_gt += len(gt_m)
            gt_cx = [r["gt_cx"][k] for k in gt_m]
            pr_m = [k for k, v in enumerate(r["pr_cls"]) if v == c]
            if not pr_m:
                continue
            pr_cx = [r["pr_cx"][k] for k in pr_m]
            pr_cf = [r["pr_cf"][k] for k in pr_m]
            M = iou_matriz(pr_cx, gt_cx)
            usados = set()
            for i in np.argsort(-np.asarray(pr_cf)):
                confs.append(float(pr_cf[i]))
                if M.size == 0:
                    tps.append(0); continue
                j = int(np.argmax(M[i])); v = float(M[i, j])
                if v >= iou_tp and j not in usados:
                    usados.add(j); tps.append(1)
                else:
                    tps.append(0)
        confs = np.asarray(confs); tps = np.asarray(tps)
        if len(confs) == 0 or n_gt == 0:
            saida[c] = dict(ap=0.0, recall=np.zeros(1), precision=np.zeros(1),
                            confs=confs, tps=tps, n_gt=n_gt)
            continue
        o = np.argsort(-confs); confs, tps = confs[o], tps[o]
        tp_ac = np.cumsum(tps); fp_ac = np.cumsum(1 - tps)
        rec = tp_ac / n_gt
        prec = tp_ac / np.maximum(tp_ac + fp_ac, 1e-9)
        ap, _, _ = _ap_todos_pontos(rec, prec)
        saida[c] = dict(ap=ap, recall=rec, precision=prec, confs=confs,
                        tps=tps, n_gt=n_gt)
    return saida

def mapa(registros, n_classes, ious=None):
    """mAP@50 e mAP@50:95 calculados a partir dos mesmos registros."""
    ious = ious if ious is not None else np.arange(0.5, 1.0, 0.05)
    por_iou = []
    for t in ious:
        c = curvas_por_classe(registros, n_classes, iou_tp=float(t))
        por_iou.append(np.mean([c[k]["ap"] for k in c]))
    return dict(map50=float(por_iou[0]), map50_95=float(np.mean(por_iou)),
                por_iou=dict(zip([round(float(t), 2) for t in ious], por_iou)))

# --------------------------------------------------------- calibração
def ece(confs, acertos, n_bins=10):
    """Expected Calibration Error + dados do diagrama de confiabilidade."""
    confs = np.asarray(confs, float); acertos = np.asarray(acertos, float)
    if len(confs) == 0:
        return 0.0, []
    bordas = np.linspace(0, 1, n_bins + 1)
    erro, linhas = 0.0, []
    for i in range(n_bins):
        m = (confs > bordas[i]) & (confs <= bordas[i + 1])
        if m.sum() == 0:
            linhas.append(dict(centro=(bordas[i]+bordas[i+1])/2, n=0,
                               conf_media=np.nan, acuracia=np.nan)); continue
        cm, ac = confs[m].mean(), acertos[m].mean()
        erro += m.sum() / len(confs) * abs(ac - cm)
        linhas.append(dict(centro=(bordas[i]+bordas[i+1])/2, n=int(m.sum()),
                           conf_media=float(cm), acuracia=float(ac)))
    return float(erro), linhas

# ------------------------------------------------- ponto de operação
def varredura_limiar(registros, n_classes, iou_tp=0.5, limiares=None):
    """P, R e F1 globais (micro) em função do limiar de confiança."""
    limiares = limiares if limiares is not None else np.round(np.arange(0.05, 0.96, 0.01), 2)
    linhas = []
    for t in limiares:
        TP = FP = FN = 0
        for r in registros:
            keep = [i for i, c in enumerate(r["pr_cf"]) if c >= t]
            pr_cx = [r["pr_cx"][i] for i in keep]
            pr_cf = [r["pr_cf"][i] for i in keep]
            pr_cls = [r["pr_cls"][i] for i in keep]
            res, falt = casar_imagem(r["gt_cx"], r["gt_cls"], pr_cx, pr_cf, pr_cls, iou_tp)
            TP += sum(1 for x in res if x[1] == "TP")
            FP += sum(1 for x in res if x[1] == "FP")
            FN += len(falt)
        P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9)
        F1 = 2 * P * R / max(P + R, 1e-9)
        linhas.append(dict(limiar=float(t), TP=TP, FP=FP, FN=FN,
                           precisao=P, recall=R, f1=F1))
    return linhas

def escolher_ponto_operacional(varredura, recall_minimo=0.85):
    """
    Duas políticas:
      - 'f1'      : maximiza F1 (equilíbrio)
      - 'triagem' : maior limiar que ainda garante recall >= recall_minimo
                    (rastreio clínico prioriza não perder lesão)
    """
    melhor_f1 = max(varredura, key=lambda l: l["f1"])
    viaveis = [l for l in varredura if l["recall"] >= recall_minimo]
    triagem = max(viaveis, key=lambda l: l["limiar"]) if viaveis else None
    return dict(f1=melhor_f1, triagem=triagem, recall_minimo=recall_minimo)

# ------------------------------------------------------ matriz de confusão
def matriz_confusao(registros, n_classes, nomes, limiar=0.25, iou_tp=0.5):
    """(n+1)x(n+1): última linha/coluna = fundo (FP de fundo / GT não detectado)."""
    n = n_classes
    M = np.zeros((n + 1, n + 1), int)   # [predito, verdadeiro]
    for r in registros:
        keep = [i for i, c in enumerate(r["pr_cf"]) if c >= limiar]
        pr_cx = [r["pr_cx"][i] for i in keep]
        pr_cf = [r["pr_cf"][i] for i in keep]
        pr_cls = [r["pr_cls"][i] for i in keep]
        Mi = iou_matriz(pr_cx, r["gt_cx"]) if pr_cx and len(r["gt_cx"]) else np.zeros((len(pr_cx), len(r["gt_cx"])))
        usados = set()
        for i in np.argsort(-np.asarray(pr_cf)) if pr_cf else []:
            if Mi.size == 0:
                M[pr_cls[i], n] += 1; continue
            j = int(np.argmax(Mi[i])); v = float(Mi[i, j])
            if v >= iou_tp and j not in usados:
                usados.add(j); M[pr_cls[i], r["gt_cls"][j]] += 1
            else:
                M[pr_cls[i], n] += 1
        for j in range(len(r["gt_cx"])):
            if j not in usados:
                M[n, r["gt_cls"][j]] += 1
    rotulos = list(nomes) + ["fundo"]
    return M, rotulos

def taxonomia_erros(registros, limiar=0.25, iou_tp=0.5):
    cont = {k: 0 for k in CATEGORIAS_ERRO}
    tp = faltas = 0
    for r in registros:
        keep = [i for i, c in enumerate(r["pr_cf"]) if c >= limiar]
        res, falt = casar_imagem(r["gt_cx"], r["gt_cls"],
                                 [r["pr_cx"][i] for i in keep],
                                 [r["pr_cf"][i] for i in keep],
                                 [r["pr_cls"][i] for i in keep], iou_tp)
        tp += sum(1 for x in res if x[1] == "TP")
        for x in res:
            if x[1] == "FP":
                cont[x[2]] += 1
        faltas += len(falt)
    cont_total = dict(cont); cont_total["nao_detectado"] = faltas
    return dict(tp=tp, erros=cont_total, total_fp=sum(cont.values()))
