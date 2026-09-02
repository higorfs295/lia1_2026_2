# =====================================================================
# CÉLULA: particionamento por protocolo de aquisição (domain split)
# =====================================================================
import os, glob, shutil, json, collections
from PIL import Image

RAIZ = "/content/datasets"          # ajustado em tempo de execução
GEOMETRIA_EXTERNA = (192, 256)      # protocolo mantido 100% fora do treino

def ler_rotulos(caminho_txt):
    if not os.path.exists(caminho_txt):
        return []
    linhas = [l.strip() for l in open(caminho_txt).read().strip().split("\n") if l.strip()]
    saida = []
    for l in linhas:
        p = l.split()
        saida.append((int(p[0]), *[float(v) for v in p[1:5]]))
    return saida

def caminho_rotulo(caminho_img):
    return os.path.splitext(caminho_img.replace("/images/", "/labels/"))[0] + ".txt"

def inventariar(raiz_ds):
    """Lê todo o dataset original e devolve uma lista de registros."""
    reg = []
    for split in ("train", "val"):
        for ip in sorted(glob.glob(f"{raiz_ds}/images/{split}/*")):
            with Image.open(ip) as im:
                w, h = im.size; modo = im.mode
            cxs = ler_rotulos(caminho_rotulo(ip))
            reg.append(dict(caminho=ip, split_original=split, largura=w, altura=h,
                            modo=modo, n_caixas=len(cxs), caixas=cxs))
    return reg

def particionar(reg, geometria_externa=GEOMETRIA_EXTERNA):
    """
    D_treino  : protocolos vistos, split train original
    D_interno : protocolos vistos, split val original  (validação in-distribution)
    D_externo : TODAS as imagens do protocolo `geometria_externa` (domain shift puro)
    """
    d = {"treino": [], "interno": [], "externo": []}
    for r in reg:
        if (r["largura"], r["altura"]) == geometria_externa:
            d["externo"].append(r)
        elif r["split_original"] == "train":
            d["treino"].append(r)
        else:
            d["interno"].append(r)
    return d

def materializar(particoes, destino, nomes=("negative", "positive")):
    """Copia arquivos para uma árvore YOLO e escreve os data.yaml."""
    os.makedirs(destino, exist_ok=True)
    for nome, regs in particoes.items():
        for sub in ("images", "labels"):
            os.makedirs(f"{destino}/{sub}/{nome}", exist_ok=True)
        for r in regs:
            base = os.path.basename(r["caminho"])
            stem = os.path.splitext(base)[0]
            shutil.copy(r["caminho"], f"{destino}/images/{nome}/{base}")
            lp = caminho_rotulo(r["caminho"])
            alvo = f"{destino}/labels/{nome}/{stem}.txt"
            if os.path.exists(lp):
                shutil.copy(lp, alvo)
            else:
                open(alvo, "w").close()          # imagem de fundo, rótulo vazio
    yamls = {}
    bloco_nomes = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(nomes))
    for val_nome in ("interno", "externo"):
        p = f"{destino}/data_{val_nome}.yaml"
        open(p, "w").write(
            f"# gerado automaticamente pelo pipeline\npath: {os.path.abspath(destino)}\n"
            f"train: images/treino\nval: images/{val_nome}\n\n{bloco_nomes}")
        yamls[val_nome] = p
    return yamls

def resumo(particoes):
    linhas = []
    for nome, regs in particoes.items():
        cls = collections.Counter()
        geo = collections.Counter()
        vazias = 0
        for r in regs:
            geo[(r["largura"], r["altura"])] += 1
            if r["n_caixas"] == 0: vazias += 1
            for c in r["caixas"]: cls[c[0]] += 1
        linhas.append(dict(particao=nome, imagens=len(regs), caixas=sum(cls.values()),
                           negative=cls[0], positive=cls[1], sem_caixa=vazias,
                           geometrias=len(geo),
                           geometria_dominante=max(geo, key=geo.get) if geo else None))
    return linhas
