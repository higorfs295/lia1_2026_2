# =====================================================================
# CÉLULA: identidade visual — extraída do template Dtox (dtox-1.0.0)
# =====================================================================
# Todas as cores abaixo vêm do próprio template:
#   scss/_variables.scss  ->  $primary-color, $text-color, $text-color-dark,
#                             $gray, $primary-gradient, $secondary-gradient
#   images/background-shape/*.png -> cores das formas decorativas
# Nada aqui foi inventado: é o sistema visual do template aplicado às figuras.
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PALETA = {
    # _variables.scss
    "primaria":  "#008DEC",   # $primary-color
    "escuro":    "#091337",   # $text-color-dark  (títulos)
    "texto":     "#4D546F",   # $text-color       (corpo)
    "cinza":     "#F2F3F5",   # $gray             (fundos de seção)
    "branco":    "#FFFFFF",   # $body-color
    # $primary-gradient / $secondary-gradient
    "turquesa":  "#17FFD3",
    "lima":      "#D3FC71",
    "ciano":     "#23E3EE",
    # formas decorativas do template
    "azul_forte":"#0070F0",
    "roxo":      "#9000F0",
    "verde":     "#10F030",
    "magenta":   "#F04090",
    "laranja":   "#F09000",
    "borda":     "#E3E7EE",
}
# ciclo de cores das séries, na ordem em que o template usa suas cores
CICLO = [PALETA["primaria"], PALETA["turquesa"], PALETA["roxo"], PALETA["laranja"],
         PALETA["magenta"], PALETA["verde"], PALETA["azul_forte"], PALETA["lima"]]

# rampa contínua reproduzindo o $secondary-gradient do template
CMAP_DTOX = LinearSegmentedColormap.from_list(
    "dtox", [PALETA["escuro"], "#0B3B7A", PALETA["primaria"],
             PALETA["ciano"], PALETA["turquesa"], PALETA["lima"]])

# a fonte do template é Poppins; se não estiver instalada, cai em DejaVu Sans
def _familia_disponivel():
    try:
        from matplotlib import font_manager
        nomes = {f.name for f in font_manager.fontManager.ttflist}
        for c in ("Poppins", "Montserrat", "DejaVu Sans"):
            if c in nomes:
                return c
    except Exception:
        pass
    return "DejaVu Sans"

FONTE = _familia_disponivel()

def aplicar_estilo():
    mpl.rcParams.update({
        "figure.facecolor": PALETA["branco"],
        "axes.facecolor":   PALETA["branco"],
        "savefig.facecolor":PALETA["branco"],
        "axes.edgecolor":   PALETA["borda"],
        "axes.labelcolor":  PALETA["texto"],
        "axes.titlecolor":  PALETA["escuro"],
        "axes.titleweight": "bold",
        "axes.titlesize":   13,
        "axes.labelsize":   10,
        "axes.grid":        True,
        "axes.axisbelow":   True,
        "grid.color":       PALETA["cinza"],
        "grid.linewidth":   1.0,
        "text.color":       PALETA["texto"],
        "xtick.color":      PALETA["texto"],
        "ytick.color":      PALETA["texto"],
        "xtick.labelsize":  9, "ytick.labelsize": 9,
        "legend.frameon":   True,
        "legend.framealpha":1.0,
        "legend.edgecolor": PALETA["borda"],
        "legend.fancybox":  True,
        "font.family":      FONTE,
        "font.weight":      "normal",
        "figure.dpi":       110,
        "savefig.dpi":      160,
        "savefig.bbox":     "tight",
        "axes.prop_cycle":  mpl.cycler(color=CICLO),
        "lines.linewidth":  2.4,
        "patch.linewidth":  0,
    })

def garantir_poppins(pasta="/tmp/fontes"):
    """
    Tenta instalar a Poppins (a fonte do template) para o matplotlib.
    Se o download falhar — sem rede, espelho fora do ar —, o projeto segue com a
    fonte de fallback e as figuras continuam corretas: é um enfeite, não um requisito.
    Devolve o nome da família que ficou ativa.
    """
    global FONTE
    from matplotlib import font_manager
    if "Poppins" in {f.name for f in font_manager.fontManager.ttflist}:
        FONTE = "Poppins"; aplicar_estilo(); return FONTE
    import os, urllib.request
    base = ("https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-%s.ttf")
    os.makedirs(pasta, exist_ok=True)
    baixadas = 0
    for peso in ("Light", "Regular", "Medium", "SemiBold", "Bold"):
        destino = os.path.join(pasta, f"Poppins-{peso}.ttf")
        try:
            if not os.path.exists(destino):
                urllib.request.urlretrieve(base % peso, destino)
            font_manager.fontManager.addfont(destino)
            baixadas += 1
        except Exception:
            pass
    FONTE = "Poppins" if baixadas else _familia_disponivel()
    aplicar_estilo()
    return FONTE

def titular(ax, titulo, subtitulo=None):
    """Título no padrão do template: h2 escuro + parágrafo de apoio."""
    ax.set_title(titulo, loc="left", pad=16 if subtitulo else 9)
    if subtitulo:
        ax.text(0, 1.015, subtitulo, transform=ax.transAxes,
                fontsize=8.5, color=PALETA["texto"], va="bottom", alpha=.85)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(PALETA["borda"])
    ax.spines["bottom"].set_color(PALETA["primaria"])
    ax.spines["bottom"].set_linewidth(1.8)
    return ax

def moldura(fig, rotulo="Projeto Final · YOLO26 · Detecção e Laudo de Tumor Cerebral"):
    fig.text(0.005, -0.02, rotulo, fontsize=7.5, color=PALETA["texto"], ha="left", alpha=.7)
    return fig
