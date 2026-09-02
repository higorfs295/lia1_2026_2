# -*- coding: utf-8 -*-
"""
Neuro26 — aplicação web do projeto de detecção e laudo de tumor cerebral.

Front-end construído sobre o template Dtox (Themefisher), servido por Flask.
O back-end reaproveita exatamente os mesmos módulos do notebook, em `pipeline/`.

    python app.py --pesos runs/yolo26_padrao/weights/best.pt \
                  --dados particionado --resultados resultados.json

Dentro do Colab a seção 8 do notebook chama `criar_app(...)` diretamente.
"""
import os, io, json, glob, argparse, datetime, tempfile
import cv2
import numpy as np
from flask import (Flask, render_template, request, send_file, url_for,
                   redirect, abort, jsonify)

# O motor é o mesmo do notebook. Quando a aplicação roda como pacote, ele vem de
# `pipeline/`; quando roda dentro do Colab, o notebook injeta as mesmas funções
# em `criar_app(..., motor=...)` — em nenhum dos casos há código duplicado.
try:
    from pipeline.inferencia import gerar_laudo, laudo_em_texto
    from pipeline.sintetico import carregar_cinza, mascara_encefalo
    MOTOR = dict(gerar_laudo=gerar_laudo, laudo_em_texto=laudo_em_texto,
                 carregar_cinza=carregar_cinza, mascara_encefalo=mascara_encefalo)
except ImportError:                                    # execução a partir do notebook
    MOTOR = {}

RAIZ = os.path.dirname(os.path.abspath(__file__))
NOMES_CLASSES = ["negative", "positive"]

# =====================================================================
# Conteúdo editorial (o mesmo do notebook e do relatório)
# =====================================================================
MOVIMENTOS = [
    dict(titulo="Entender o YOLO26", destaque=False,
         texto="Cabeça one-to-one sem NMS, sem DFL, com STAL e ProgLoss. Isso muda o código: "
               "augment=True deixa de existir e a incerteza precisa vir de outro lugar."),
    dict(titulo="Ler o dado cru", destaque=False,
         texto="Abrir cada imagem e cada arquivo de rótulo. É essa leitura que revela vários "
               "protocolos de aquisição misturados no mesmo split."),
    dict(titulo="Particionar por protocolo", destaque=True,
         texto="Todo o protocolo 192×256 fica fora do treino. Holdout de protocolo, "
               "não holdout aleatório — a versão honesta da pergunta."),
    dict(titulo="Sintetizar lesões", destaque=True,
         texto="Copy-paste com máscara do encéfalo, harmonização de intensidade e blending de "
               "Poisson. O rótulo é conhecido, não estimado."),
    dict(titulo="Treinar sob orçamento", destaque=False,
         texto="O argumento time= sobrescreve epochs: as épocas viram saída do experimento "
               "e o relógio fica fixo em qualquer GPU."),
    dict(titulo="Diagnosticar o modelo", destaque=False,
         texto="Calibração, limiar operacional justificado, taxonomia de erros, domain shift, "
               "robustez e alarme falso fora do domínio."),
    dict(titulo="Entregar o laudo", destaque=False,
         texto="Tensor vira JSON versionado, e JSON vira esta interface — usável sem abrir "
               "o notebook."),
]

PARTICOES = [
    dict(nome="treino",  papel="ajuste dos pesos", imagens=718, caixas=752,
         negative=409, positive=343, protocolo="512² · 256² · raros", destaque=False),
    dict(nome="interno", papel="validação in-distribution", imagens=223, caixas=241,
         negative=154, positive=87, protocolo="512² · 256²", destaque=False),
    dict(nome="externo", papel="validação sob domain shift", imagens=175, caixas=173,
         negative=28, positive=145, protocolo="192×256 exclusivo", destaque=True),
]

SINTESE_CURTA = [
    dict(icone="ti-target", titulo="Onde colar",
         texto="Máscara intracraniana por Otsu, maior componente conexo e fechamento morfológico."),
    dict(icone="ti-blend", titulo="Como misturar",
         texto="Máscara elíptica com borda difusa, para a lesão desaparecer no tecido."),
    dict(icone="ti-shine", titulo="Harmonizar",
         texto="Casamento de média e desvio com a vizinhança do destino."),
    dict(icone="ti-layers", titulo="Ou Poisson",
         texto="Seamless cloning: a costura some porque a solução impõe continuidade no contorno."),
]

SINTESE_LONGA = [
    dict(titulo="Onde colar",
         texto="Máscara intracraniana obtida por limiarização de Otsu, maior componente conexo, "
               "fechamento morfológico e preenchimento de contorno; depois uma erosão afasta a "
               "colagem da calota craniana.",
         formula="M = preenche((B ⊕ K) ⊖ K), depois erodida"),
    dict(titulo="Como misturar",
         texto="Máscara elíptica com borda difusa: para a distância normalizada d do centro, a "
               "lesão se dissolve gradualmente no tecido em vez de terminar numa aresta.",
         formula="α = clip((1 − d)/s, 0, 1);  I = α·I_lesão + (1−α)·I_fundo"),
    dict(titulo="Harmonizar intensidade",
         texto="O recorte vem de outra imagem, com outro brilho. Casam-se os dois primeiros "
               "momentos com a vizinhança do destino antes de misturar.",
         formula="I′ = (I − μ_lesão)/σ_lesão · σ_viz + μ_viz"),
    dict(titulo="Ou resolver Poisson",
         texto="Em parte das amostras usa-se seamless cloning: a costura desaparece porque a "
               "solução impõe continuidade no contorno da região colada.",
         formula="∇²f = ∇²g em Ω,  f|∂Ω = I_fundo|∂Ω"),
]

YOLO26 = [
    dict(titulo="Cabeça end-to-end",
         texto="Uma caixa por objeto já na saída da rede, sem supressão de não-máximos. "
               "Os parâmetros iou= e agnostic_nms= perdem o efeito de antes."),
    dict(titulo="Sem DFL",
         texto="Remover a Distribution Focal Loss simplifica a cabeça e mantém a regressão sem "
               "faixa limitada — o que ajuda em alvos pequenos, que é o nosso caso."),
    dict(titulo="STAL e ProgLoss",
         texto="Atribuição de rótulos ciente de alvos pequenos e perda progressiva. A mediana das "
               "lesões aqui ocupa ~1,5 % da área da imagem."),
]

TREINO = [
    dict(arg="time",          valor="0,25 h", motivo="impõe o teto de 15 min; sobrescreve epochs"),
    dict(arg="epochs",        valor="120",    motivo="apenas um teto superior; quem decide é time"),
    dict(arg="imgsz",         valor="640",    motivo="compromisso entre alvo pequeno e velocidade"),
    dict(arg="cache",         valor="ram",    motivo="≈1.000 imagens pequenas; o gargalo é o disco"),
    dict(arg="cos_lr",        valor="True",   motivo="decaimento suave, robusto a número de épocas indeterminado"),
    dict(arg="close_mosaic",  valor="5",      motivo="o modelo vê imagens inteiras antes de encerrar"),
    dict(arg="deterministic", valor="False",  motivo="reprodutibilidade total custaria velocidade — assunção declarada"),
    dict(arg="patience",      valor="100",    motivo="o critério de parada é o tempo, não a estagnação"),
]

EIXOS = [
    dict(icone="ti-bar-chart", pergunta="Quão bem, no total?",
         instrumento="mAP@50 e mAP@50-95 recalculados a partir das predições cruas."),
    dict(icone="ti-control-shuffle", pergunta="Com qual limiar operar?",
         instrumento="Varredura de τ, curva precisão-recall e duas políticas de escolha."),
    dict(icone="ti-ruler-alt-2", pergunta="A confiança significa algo?",
         instrumento="ECE e diagrama de confiabilidade sobre as detecções."),
    dict(icone="ti-search", pergunta="Que tipo de erro ele comete?",
         instrumento="Taxonomia inspirada no TIDE e matriz de confusão com fundo."),
    dict(icone="ti-exchange-vertical", pergunta="Sobrevive a outro aparelho?",
         instrumento="Partição externa por protocolo e artefatos de RM simulados."),
    dict(icone="ti-alert", pergunta="Inventa achado onde não há nada?",
         instrumento="Controle negativo com 115 imagens de outro domínio médico."),
]

LIMITES = [
    dict(titulo="Não é diagnóstico.",
         texto="As classes são convenção de anotação de um conjunto público, sem laudo clínico "
               "verificado. O sistema aprende a convenção, não a doença."),
    dict(titulo="A partição externa é um proxy.",
         texto="Geometria de aquisição sugere fonte diferente, mas não prova. A afirmação "
               "defensável é «há deslocamento mensurável entre esses subconjuntos»."),
    dict(titulo="O sintético herda o viés do original.",
         texto="As lesões vêm do mesmo banco de 745 recortes: ampliam variedade de posição e "
               "escala, não criam patologia nova."),
    dict(titulo="Os artefatos são simulados.",
         texto="Reproduzem o modelo físico, não a distribuição empírica de um equipamento real."),
    dict(titulo="Estabilidade não é probabilidade.",
         texto="É concordância entre três vistas da mesma imagem; um modelo consistentemente "
               "errado exibe estabilidade alta."),
    dict(titulo="O orçamento limita o resultado.",
         texto="Com quinze minutos de treino, o número reportado é um piso, não o teto do modelo."),
    dict(titulo="Não há escala física.",
         texto="Sem o campo PixelSpacing do DICOM, nenhuma medida é reportada em milímetros."),
]

REFERENCIAS = [
    "REDMON, J. et al. You Only Look Once: Unified, Real-Time Object Detection. arXiv:1506.02640, 2015.",
    "ULTRALYTICS. YOLO26 — Unified Real-Time End-to-End Vision Models. docs.ultralytics.com/models/yolo26.",
    "ULTRALYTICS. Brain Tumor Dataset · Train Mode. docs.ultralytics.com.",
    "BOLYA, D. et al. TIDE: A General Toolbox for Identifying Object Detection Errors. ECCV, 2020.",
    "GUO, C. et al. On Calibration of Modern Neural Networks. ICML, 2017.",
    "PÉREZ, P.; GANGNET, M.; BLAKE, A. Poisson Image Editing. ACM SIGGRAPH, 2003.",
    "GUDBJARTSSON, H.; PATZ, S. The Rician Distribution of Noisy MRI Data. MRM, v. 34, n. 6, 1995.",
    "THEMEFISHER. Dtox — template HTML utilizado como base desta interface.",
]

# cores do template, usadas nas barras do painel de resultados
GRAD_PRIMARIO = "linear-gradient(25deg,#17ffd3 0%,#d3fc71 95%)"
GRAD_SECUNDARIO = "linear-gradient(6deg,#17ffd3 0%,#23e3ee 100%)"
COR_PRIMARIA, COR_MAGENTA, COR_ROXO = "#008dec", "#f04090", "#9000f0"


# =====================================================================
# figuras que as páginas sabem exibir: chave -> nome do arquivo padrão
FIGURAS_PADRAO = {
    "pipeline":   "f0_pipeline.png",
    "composicao": "f1_composicao.png",
    "geometria":  "f2_geometria.png",
    "calor":      "f3_calor.png",
    "sinteticas": "f4_sinteticas.png",
    "artefatos":  "f5_artefatos.png",
}
# figuras de diagnóstico, exibidas na página de resultados quando existirem
FIGURAS_DIAGNOSTICO = [
    ("pr_limiar",   "Curva precisão-recall e escolha do limiar operacional"),
    ("calibracao",  "Diagrama de confiabilidade e massa por faixa de confiança"),
    ("matriz",      "Matriz de confusão no limiar adotado"),
    ("taxonomia",   "Anatomia do erro: onde o modelo falha"),
    ("dominios",    "Generalização entre domínios"),
    ("robustez",    "Degradação sob artefatos de ressonância"),
    ("qualitativa", "Acertos, falsos positivos e lesões perdidas"),
    ("treino",      "Perdas e métricas ao longo do treino"),
    ("orcamento",   "Para onde foram os minutos"),
]


def criar_app(pesos=None, dados=None, resultados=None, imgsz=640, device=None,
              saida=None, figuras=None, motor=None):
    if motor:
        MOTOR.update(motor)
    faltando = [k for k in ("gerar_laudo", "carregar_cinza", "mascara_encefalo")
                if k not in MOTOR]
    if faltando:
        raise RuntimeError("motor incompleto, faltam: " + ", ".join(faltando))

    app = Flask(__name__, template_folder=os.path.join(RAIZ, "templates"),
                static_folder=os.path.join(RAIZ, "static"))
    app.config["MAX_CONTENT_LENGTH"] = 24 * 1024 * 1024

    # modo completo = o pacote original do Dtox está presente em static/dtox
    completo = os.path.exists(os.path.join(RAIZ, "static", "dtox", "css", "style.css"))
    app.jinja_env.globals["completo"] = completo

    est = dict(modelo=None, pesos=pesos, dados=dados, imgsz=imgsz, device=device,
               resultados=None, caminho_resultados=resultados,
               saida=saida or os.path.join(RAIZ, "instancia"))
    os.makedirs(est["saida"], exist_ok=True)

    # figuras: as passadas pelo notebook têm prioridade; senão, as do pacote
    pasta_fig = os.path.join(RAIZ, "static", "projeto", "figuras")
    os.makedirs(pasta_fig, exist_ok=True)
    mapa_fig = {}
    for chave, arq in FIGURAS_PADRAO.items():
        if os.path.exists(os.path.join(pasta_fig, arq)):
            mapa_fig[chave] = arq
    if figuras:
        import shutil as _sh
        for chave, caminho in figuras.items():
            if caminho and os.path.exists(caminho):
                destino = f"{chave}.png"
                _sh.copy(caminho, os.path.join(pasta_fig, destino))
                mapa_fig[chave] = destino
    app.jinja_env.globals["figuras"] = mapa_fig
    app.jinja_env.globals["figuras_diagnostico"] = [
        (c, t, mapa_fig[c]) for c, t in FIGURAS_DIAGNOSTICO if c in mapa_fig]

    if resultados and os.path.exists(resultados):
        with open(resultados, encoding="utf-8") as f:
            est["resultados"] = json.load(f)

    def modelo():
        if est["modelo"] is None and est["pesos"] and os.path.exists(est["pesos"]):
            from ultralytics import YOLO
            est["modelo"] = YOLO(est["pesos"])
        return est["modelo"]

    def exemplos():
        if not est["dados"]:
            return []
        alvo = os.path.join(est["dados"], "images", "interno")
        return [os.path.basename(p) for p in sorted(glob.glob(alvo + "/*"))[:8]]

    # ---------------------------------------------------------- rotas
    @app.route("/")
    def inicio():
        return render_template("index.html", movimentos=MOVIMENTOS,
                               particoes=PARTICOES, sintese=SINTESE_CURTA)

    @app.route("/metodo")
    def metodo():
        return render_template("metodo.html", yolo=YOLO26, sintese=SINTESE_LONGA,
                               treino=TREINO, eixos=EIXOS, limites=LIMITES,
                               referencias=REFERENCIAS)

    @app.route("/miniatura/<path:nome>")
    def miniatura(nome):
        cam = os.path.join(est["dados"] or "", "images", "interno", nome)
        if not os.path.exists(cam):
            abort(404)
        img = MOTOR["carregar_cinza"](cam)
        # recorta no encéfalo para a miniatura não virar um quadrado preto
        try:
            ys, xs = np.where(MOTOR["mascara_encefalo"](img) > 0)
            if len(xs):
                m = int(0.05 * max(img.shape))
                img = img[max(0, ys.min() - m):ys.max() + m,
                          max(0, xs.min() - m):xs.max() + m]
        except Exception:
            pass
        img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")

    @app.route("/saida/<path:nome>")
    def arquivo_saida(nome):
        cam = os.path.join(est["saida"], nome)
        if not os.path.exists(cam):
            abort(404)
        return send_file(cam)

    @app.route("/laudo/<path:nome>")
    def baixar_laudo(nome):
        cam = os.path.join(est["saida"], nome)
        if not os.path.exists(cam):
            abort(404)
        return send_file(cam, as_attachment=True, download_name=nome)

    def _desenhar(img_cinza, laudo, destino):
        """Anota as caixas usando as cores do template."""
        vis = cv2.cvtColor(img_cinza, cv2.COLOR_GRAY2BGR)
        cores = {"positive": (240, 0, 144), "negative": (236, 141, 0)}  # BGR
        for d in laudo["achados"]:
            x1, y1, x2, y2 = [int(v) for v in d["caixa_px"]]
            c = cores.get(d["achado"], (236, 141, 0))
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            rot = f"{d['id']} {d['achado']} {d['confianca']:.2f}"
            (tw, th), _ = cv2.getTextSize(rot, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), c, -1)
            cv2.putText(vis, rot, (x1 + 4, max(11, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        # enquadra no encéfalo: o fundo preto não carrega informação
        try:
            ys, xs = np.where(MOTOR["mascara_encefalo"](img_cinza) > 0)
            if len(xs):
                m = int(0.06 * max(img_cinza.shape))
                vis = vis[max(0, ys.min() - m):min(vis.shape[0], ys.max() + m),
                          max(0, xs.min() - m):min(vis.shape[1], xs.max() + m)]
        except Exception:
            pass
        cv2.imwrite(destino, vis)

    @app.route("/laudo", methods=["GET", "POST"])
    def laudo():
        ctx = dict(tau=0.25, iou=0.55, escala=None, consistencia=True,
                   exemplos=exemplos(), exemplo_ativo=None, laudo=None,
                   modelo_pronto=bool(est["pesos"] and os.path.exists(est["pesos"])))
        if request.method == "GET":
            return render_template("laudo.html", **ctx)

        ctx["tau"] = float(request.form.get("tau", 0.25))
        ctx["iou"] = float(request.form.get("iou", 0.55) or 0.55)
        ctx["consistencia"] = bool(request.form.get("consistencia"))
        escala = request.form.get("escala") or None
        ctx["escala"] = float(escala) if escala else None

        arquivo = request.files.get("imagem")
        escolhido = request.form.get("exemplo")
        entrada = None
        if arquivo and arquivo.filename:
            dados = np.frombuffer(arquivo.read(), np.uint8)
            img = cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return render_template("laudo.html", **ctx)
            entrada = os.path.join(est["saida"], "entrada.png")
            cv2.imwrite(entrada, img)
        elif escolhido:
            ctx["exemplo_ativo"] = escolhido
            entrada = os.path.join(est["dados"], "images", "interno", escolhido)
        if entrada is None or modelo() is None:
            return render_template("laudo.html", **ctx)

        laudo_, _ = MOTOR["gerar_laudo"](modelo(), entrada, NOMES_CLASSES, limiar=ctx["tau"],
                                imgsz=est["imgsz"], iou_consolidacao=ctx["iou"],
                                device=est["device"], consistencia=ctx["consistencia"],
                                escala_mm_por_px=ctx["escala"])
        carimbo = datetime.datetime.now().strftime("%H%M%S")
        anotada = f"anotada_{carimbo}.png"
        _desenhar(MOTOR["carregar_cinza"](entrada), laudo_, os.path.join(est["saida"], anotada))
        json_nome = f"laudo_{carimbo}.json"
        with open(os.path.join(est["saida"], json_nome), "w", encoding="utf-8") as f:
            json.dump(laudo_, f, ensure_ascii=False, indent=2)
        ctx.update(laudo=laudo_, anotada=anotada, json_nome=json_nome)
        return render_template("laudo.html", **ctx)

    @app.route("/api/laudo", methods=["POST"])
    def api_laudo():
        """Mesmo motor, resposta em JSON — para integrar com outro sistema."""
        if modelo() is None:
            return jsonify(erro="modelo não carregado"), 503
        arquivo = request.files.get("imagem")
        if not arquivo:
            return jsonify(erro="envie o campo 'imagem'"), 400
        img = cv2.imdecode(np.frombuffer(arquivo.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify(erro="não consegui decodificar a imagem"), 400
        tmp = os.path.join(est["saida"], "api_entrada.png")
        cv2.imwrite(tmp, img)
        laudo_, _ = MOTOR["gerar_laudo"](modelo(), tmp, NOMES_CLASSES,
                                limiar=float(request.form.get("tau", 0.25)),
                                imgsz=est["imgsz"], device=est["device"],
                                consistencia=request.form.get("consistencia", "1") == "1")
        return jsonify(laudo_)

    # ----------------------------------------------------- resultados
    def montar_resultados(d):
        m, t = d.get("metricas", {}), d.get("tempo", {})
        cfg, ood = d.get("config", {}), d.get("ood", {})
        f = lambda v, n=3: ("%." + str(n) + "f") % float(v)
        kpis = [
            dict(rotulo="mAP@50 interno",  valor=f(m["interno"]["map50"]),
                 apoio="in-distribution", acento=True),
            dict(rotulo="mAP@50 externo",  valor=f(m["externo"]["map50"]),
                 apoio="sob domain shift", acento=True),
            dict(rotulo="Queda de domínio", valor=f(m.get("queda_mAP50_dominio_pct", 0), 1) + "%",
                 apoio="interno → externo", acento=False),
            dict(rotulo="ECE", valor=f(m.get("ece", 0)),
                 apoio="0 = perfeitamente calibrado", acento=False),
            dict(rotulo="Limiar operacional", valor=f(m.get("tau_operacional", 0), 2),
                 apoio="τ adotado no laudo", acento=False),
            dict(rotulo="Alarme falso OOD", valor=f(ood.get("taxa_alarme_falso", 0)),
                 apoio="medical-pills", acento=False),
            dict(rotulo="Tempo total", valor=f(t.get("total_min", 0), 1) + " min",
                 apoio="teto de %s min" % t.get("orcamento_min", 30), acento=False),
            dict(rotulo="Modelo", valor=str(cfg.get("modelo", "—")).replace(".pt", ""),
                 apoio="perfil %s · %s época(s)" % (d.get("perfil", "—"),
                       d.get("treino", {}).get("epocas_concluidas", "—")), acento=False),
        ]
        pares = [("interno · mAP@50", m["interno"]["map50"], GRAD_PRIMARIO),
                 ("externo · mAP@50", m["externo"]["map50"], GRAD_SECUNDARIO),
                 ("interno · mAP@50-95", m["interno"]["map50_95"], GRAD_PRIMARIO),
                 ("externo · mAP@50-95", m["externo"]["map50_95"], GRAD_SECUNDARIO)]
        barras = [dict(rotulo=r, texto=f(v), pct=round(100 * float(v), 1), gradiente=g)
                  for r, v, g in pares]

        rot = {"classificacao": "classe errada (IoU ok)", "localizacao": "caixa mal posicionada",
               "cls+loc": "classe e caixa erradas", "duplicata": "duplicata",
               "fundo": "invenção sobre o fundo", "nao_detectado": "lesão não detectada"}
        e = (d.get("erros") or {}).get("erros", {})
        mx = max([1] + [v for v in e.values()])
        erros = [dict(rotulo=rot.get(k, k), valor=v, pct=round(100 * v / mx, 1),
                      cor=COR_MAGENTA if k == "nao_detectado" else COR_PRIMARIA)
                 for k, v in sorted(e.items(), key=lambda x: -x[1])]

        rb = d.get("robustez") or {}
        sev = sorted({int(s) for v in rb.values() for s in v})
        robustez = []
        for nome, serie in rb.items():
            vals = [float(serie[str(s)]) if str(s) in serie else float(serie.get(s, 0)) for s in sev]
            base = vals[0] if vals else 0
            pior = min(vals) if vals else 0
            robustez.append(dict(nome=nome.replace("_", " "),
                                 valores=[f(v) for v in vals],
                                 queda=(f(100 * (1 - pior / base), 1) + "%") if base else "—"))

        et = sorted(t.get("etapas", []), key=lambda x: -x["segundos"])[:8]
        tmax = max([1e-9] + [x["segundos"] for x in et])
        tempos = [dict(etapa=x["etapa"], texto=f(x["segundos"] / 60, 1) + " min",
                       pct=round(100 * x["segundos"] / tmax, 1),
                       cor=COR_ROXO if "reino" in x["etapa"] else COR_PRIMARIA) for x in et]

        config = [dict(k=k, v=str(v)) for k, v in [
            ("perfil", d.get("perfil", "—")),
            ("modelo", cfg.get("modelo", "—")),
            ("imgsz", cfg.get("imgsz", "—")),
            ("orçamento de treino", str(cfg.get("orcamento_treino_min", "—")) + " min"),
            ("imagens sintéticas", d.get("dataset", {}).get("sinteticas", "—")),
            ("controle OOD", d.get("dataset", {}).get("ood_medical_pills", "—")),
            ("épocas concluídas", d.get("treino", {}).get("epocas_concluidas", "—")),
        ]]
        return dict(kpis=kpis, barras_dominio=barras, erros=erros, robustez=robustez,
                    severidades=sev, tempos=tempos, config=config)

    @app.route("/resultados", methods=["GET", "POST"])
    def resultados_():
        # o notebook grava o resultados.json depois de subir o servidor, então
        # a leitura é tentada a cada visita enquanto o arquivo não existir
        if est["resultados"] is None and est["caminho_resultados"] \
                and os.path.exists(est["caminho_resultados"]):
            try:
                with open(est["caminho_resultados"], encoding="utf-8") as f:
                    est["resultados"] = json.load(f)
            except (OSError, ValueError):
                pass
        d = est["resultados"]
        if request.method == "POST":
            arq = request.files.get("arquivo")
            if arq:
                try:
                    d = json.loads(arq.read().decode("utf-8"))
                    est["resultados"] = d
                except Exception:
                    d = None
        if not d:
            return render_template("resultados.html", dados=None)
        try:
            return render_template("resultados.html", dados=d, **montar_resultados(d))
        except (KeyError, TypeError, ValueError):
            return render_template("resultados.html", dados=None)

    app.add_url_rule("/resultados", "resultados", resultados_,
                     methods=["GET", "POST"])
    app.jinja_env.globals["agora"] = datetime.datetime.now
    app.estado = est
    app.completo = completo
    return app


# =====================================================================
def principal():
    p = argparse.ArgumentParser(description="Neuro26 — interface web do projeto")
    p.add_argument("--pesos", default=os.environ.get("NEURO26_PESOS"))
    p.add_argument("--dados", default=os.environ.get("NEURO26_DADOS"))
    p.add_argument("--resultados", default=os.environ.get("NEURO26_RESULTADOS"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--porta", type=int, default=5000)
    a = p.parse_args()
    app = criar_app(a.pesos, a.dados, a.resultados, a.imgsz, a.device)
    print(f"\n  Neuro26 no ar em http://127.0.0.1:{a.porta}"
          f"   (template Dtox {'completo' if app.completo else 'em modo enxuto'})\n")
    app.run(host="0.0.0.0", port=a.porta, debug=False)


if __name__ == "__main__":
    principal()
