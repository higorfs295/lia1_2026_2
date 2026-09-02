# =====================================================================
# CÉLULA: síntese de dados — copy-paste de lesões + artefatos de RM
# =====================================================================
import os, glob, math, random
import numpy as np
import cv2

# ---------------------------------------------------------------- utilidades
def carregar_cinza(caminho):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(caminho)
    return img

def yolo_para_pixel(cx, cy, bw, bh, W, H):
    x1 = (cx - bw / 2) * W; y1 = (cy - bh / 2) * H
    x2 = (cx + bw / 2) * W; y2 = (cy + bh / 2) * H
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def pixel_para_yolo(x1, y1, x2, y2, W, H):
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

# ------------------------------------------------- segmentação do encéfalo
def mascara_encefalo(img, fechamento=9):
    """
    Máscara grosseira da região intracraniana: Otsu -> maior componente conexo
    -> fechamento morfológico -> preenchimento de buracos.
    Serve apenas para restringir ONDE uma lesão sintética pode ser colada.
    """
    borrada = cv2.GaussianBlur(img, (5, 5), 0)
    _, bin_ = cv2.threshold(borrada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fechamento, fechamento))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_, 8)
    if n <= 1:
        return np.ones_like(img, np.uint8) * 255
    maior = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = np.where(lab == maior, 255, 0).astype(np.uint8)
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cheia = np.zeros_like(m)
    cv2.drawContours(cheia, contornos, -1, 255, cv2.FILLED)
    return cheia

# ------------------------------------------------------- banco de lesões
def banco_de_lesoes(registros, margem=0.15, lado_min=16):
    """Recorta cada caixa anotada; guarda o recorte e a classe de origem."""
    banco = []
    for r in registros:
        if not r["caixas"]:
            continue
        img = carregar_cinza(r["caminho"])
        H, W = img.shape
        for (c, cx, cy, bw, bh) in r["caixas"]:
            x1, y1, x2, y2 = yolo_para_pixel(cx, cy, bw, bh, W, H)
            mx, my = int((x2-x1) * margem), int((y2-y1) * margem)
            x1, y1 = max(0, x1-mx), max(0, y1-my)
            x2, y2 = min(W, x2+mx), min(H, y2+my)
            if x2-x1 < lado_min or y2-y1 < lado_min:
                continue
            banco.append(dict(classe=c, recorte=img[y1:y2, x1:x2].copy(),
                              origem=os.path.basename(r["caminho"])))
    return banco

def _mascara_suave(h, w, suavidade=0.30):
    """Máscara elíptica com borda difusa (feathering) — evita costura visível."""
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    d = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    m = np.clip((1.0 - d) / max(suavidade, 1e-6), 0, 1)
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigmaX=max(h, w) * 0.05)

def _casar_intensidade(recorte, vizinhanca, peso=0.7):
    """Casa média/desvio do recorte com a vizinhança do destino (harmonização)."""
    a = recorte.astype(np.float32)
    mu_a, sd_a = a.mean(), a.std() + 1e-6
    mu_b, sd_b = float(vizinhanca.mean()), float(vizinhanca.std()) + 1e-6
    ajustado = (a - mu_a) / sd_a * sd_b + mu_b
    return np.clip(peso * ajustado + (1 - peso) * a, 0, 255)

def colar_lesao(img, caixas_existentes, banco, rng, escala=(0.7, 1.3),
                tentativas=40, modo="alfa"):
    """
    Insere uma lesão do banco numa posição plausível (dentro do encéfalo,
    sem sobrepor caixas já existentes). Devolve (img_nova, caixa_nova) ou None.
    """
    H, W = img.shape
    masc = mascara_encefalo(img)
    # erosão para não colar rente à calota craniana
    er = cv2.erode(masc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    ys, xs = np.where(er > 0)
    if len(xs) == 0:
        return None
    for _ in range(tentativas):
        item = banco[rng.randrange(len(banco))]
        rec = item["recorte"]
        s = rng.uniform(*escala)
        h = max(12, int(rec.shape[0] * s)); w = max(12, int(rec.shape[1] * s))
        if h >= H * 0.6 or w >= W * 0.6:
            continue
        rec_r = cv2.resize(rec, (w, h), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.5:
            rec_r = cv2.flip(rec_r, 1)
        i = rng.randrange(len(xs))
        cx, cy = int(xs[i]), int(ys[i])
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = x1 + w, y1 + h
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            continue
        # a lesão precisa cair majoritariamente dentro do encéfalo
        if er[y1:y2, x1:x2].mean() < 200:
            continue
        nova = (x1, y1, x2, y2)
        if any(iou(nova, c) > 0.02 for c in caixas_existentes):
            continue
        destino = img.copy()
        viz = img[max(0, y1-8):min(H, y2+8), max(0, x1-8):min(W, x2+8)]
        rec_h = _casar_intensidade(rec_r, viz)
        if modo == "poisson":
            src = cv2.cvtColor(rec_h.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            dst = cv2.cvtColor(destino, cv2.COLOR_GRAY2BGR)
            centro = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            mk = (_mascara_suave(h, w) > 0.35).astype(np.uint8) * 255
            fundido = cv2.seamlessClone(src, dst, mk, centro, cv2.MIXED_CLONE)
            destino = cv2.cvtColor(fundido, cv2.COLOR_BGR2GRAY)
        else:
            alfa = _mascara_suave(h, w)[..., None][:, :, 0]
            regiao = destino[y1:y2, x1:x2].astype(np.float32)
            destino[y1:y2, x1:x2] = np.clip(alfa * rec_h + (1 - alfa) * regiao,
                                            0, 255).astype(np.uint8)
        return destino, (item["classe"], *nova)
    return None

def gerar_conjunto_sintetico(registros_treino, destino_img, destino_lab, banco,
                             n_alvo=300, semente=42, prob_poisson=0.35,
                             max_lesoes=2, prefixo="sint"):
    """Gera n_alvo imagens sintéticas com rótulos YOLO exatos."""
    rng = random.Random(semente)
    os.makedirs(destino_img, exist_ok=True); os.makedirs(destino_lab, exist_ok=True)
    base = [r for r in registros_treino]
    produzidas, tentativas, i = [], 0, 0
    while len(produzidas) < n_alvo and tentativas < n_alvo * 6:
        tentativas += 1
        r = base[rng.randrange(len(base))]
        img = carregar_cinza(r["caminho"])
        H, W = img.shape
        caixas_px = [yolo_para_pixel(cx, cy, bw, bh, W, H) for (_, cx, cy, bw, bh) in r["caixas"]]
        classes = [c for (c, *_ ) in r["caixas"]]
        novas = []
        k = rng.randint(1, max_lesoes)
        atual = img
        for _ in range(k):
            modo = "poisson" if rng.random() < prob_poisson else "alfa"
            saida = colar_lesao(atual, caixas_px + [n[1:] for n in novas], banco, rng, modo=modo)
            if saida is None:
                break
            atual, cx_nova = saida
            novas.append(cx_nova)
            caixas_px.append(cx_nova[1:])
        if not novas:
            continue
        i += 1
        nome = f"{prefixo}_{i:04d}"
        cv2.imwrite(f"{destino_img}/{nome}.jpg", atual, [cv2.IMWRITE_JPEG_QUALITY, 95])
        linhas = []
        for c, (cx, cy, bw, bh) in zip(classes, [pixel_para_yolo(*b, W, H) for b in caixas_px[:len(classes)]]):
            linhas.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        for (c, x1, y1, x2, y2) in novas:
            cx, cy, bw, bh = pixel_para_yolo(x1, y1, x2, y2, W, H)
            linhas.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        open(f"{destino_lab}/{nome}.txt", "w").write("\n".join(linhas) + "\n")
        produzidas.append(dict(nome=nome, origem=os.path.basename(r["caminho"]),
                               n_lesoes_novas=len(novas), n_caixas=len(linhas)))
    return produzidas

# =====================================================================
# Artefatos de ressonância magnética (apenas para o teste de robustez)
# =====================================================================
def ruido_riciano(img, sigma):
    """Magnitude de ruído gaussiano complexo: |(I+n1) + i·n2| — modelo correto de RM."""
    rng = np.random.default_rng(0)
    a = img.astype(np.float32)
    n1 = rng.normal(0, sigma, a.shape); n2 = rng.normal(0, sigma, a.shape)
    return np.clip(np.sqrt((a + n1) ** 2 + n2 ** 2), 0, 255).astype(np.uint8)

def campo_de_bias(img, amplitude):
    """Não-uniformidade multiplicativa suave (inomogeneidade de B1)."""
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xx = xx / W - 0.5; yy = yy / H - 0.5
    campo = 1.0 + amplitude * (0.9 * xx + 0.6 * yy + 1.4 * (xx ** 2 - yy ** 2) - 0.5)
    return np.clip(img.astype(np.float32) * campo, 0, 255).astype(np.uint8)

def ghosting(img, intensidade, periodo=2):
    """Ghosting N/2: modulação de fase em linhas alternadas do espaço-k."""
    F = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
    fase = np.ones(F.shape, dtype=np.complex64)
    fase[::periodo, :] = np.exp(1j * np.pi * intensidade)
    return np.clip(np.abs(np.fft.ifft2(np.fft.ifftshift(F * fase))), 0, 255).astype(np.uint8)

def gibbs(img, fracao_mantida):
    """Truncamento do espaço-k -> ringing de Gibbs nas bordas de alto contraste."""
    F = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
    H, W = img.shape
    mh, mw = int(H * fracao_mantida / 2), int(W * fracao_mantida / 2)
    masc = np.zeros_like(F, dtype=np.float32)
    masc[H//2-mh:H//2+mh, W//2-mw:W//2+mw] = 1.0
    return np.clip(np.abs(np.fft.ifft2(np.fft.ifftshift(F * masc))), 0, 255).astype(np.uint8)

# severidade 1..3 por artefato (parâmetros calibrados visualmente)
ARTEFATOS = {
    "ruido_riciano": (ruido_riciano, {1: 8,    2: 16,   3: 28}),
    "campo_de_bias": (campo_de_bias, {1: 0.35, 2: 0.65, 3: 1.00}),
    "ghosting":      (ghosting,      {1: 0.25, 2: 0.50, 3: 0.85}),
    "gibbs":         (gibbs,         {1: 0.40, 2: 0.24, 3: 0.14}),
}

def aplicar_artefato(img, nome, severidade):
    fn, tabela = ARTEFATOS[nome]
    return fn(img, tabela[severidade])
