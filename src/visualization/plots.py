"""Identidade visual única e utilitários de gráficos (Matplotlib + Seaborn).

Regras aplicadas em todos os gráficos do projeto:
- paleta categórica fixa por entidade (Máquina 1 = azul, Máquina 2 = laranja, Total = verde-água),
  validada para daltonismo (ΔE CVD ≥ 9 entre pares adjacentes); a cor segue a entidade, nunca o posto;
- observado sempre em tinta escura; previsão na cor da série/modelo, tracejada;
- grades discretas, eixos recessivos, títulos à esquerda, unidades nos eixos;
- quebras estruturais anotadas com linha vertical e rótulo;
- legenda sempre presente com 2+ séries (a identidade nunca depende só da cor);
- exportação em PNG (300 dpi) e PDF.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from src.settings import carregar_config

_CFG = carregar_config()["visualizacao"]
CORES: dict[str, str] = dict(_CFG["cores"])
CORES_MODELOS: dict[str, str] = dict(_CFG["cores_modelos"])
MESES_PT = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
DIAS_PT = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
SEQUENCIAL = sns.light_palette(CORES["Máquina 1"], as_cmap=True)
DIVERGENTE = sns.diverging_palette(250, 15, s=75, l=50, center="light", as_cmap=True)


def aplicar_tema() -> None:
    """Aplica o tema visual do projeto (chamar uma vez por processo/notebook)."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": _CFG["dpi"], "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 10, "axes.labelcolor": CORES["texto_secundario"],
        "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.color": CORES["texto_secundario"],
        "ytick.color": CORES["texto_secundario"], "legend.fontsize": 9, "legend.frameon": False,
        "axes.edgecolor": CORES["eixo"], "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": CORES["grade"], "grid.linewidth": 0.8, "lines.linewidth": 2.0,
        "figure.titlesize": 13, "figure.titleweight": "bold",
    })


aplicar_tema()


def cor_serie(nome: str) -> str:
    """Cor fixa de uma série (máquina/total) ou de um modelo."""
    return CORES.get(nome) or CORES_MODELOS.get(nome) or CORES["observado"]


def salvar_figura(fig: plt.Figure, pasta: str | Path, nome: str, formatos: Iterable[str] | None = None,
                  fechar: bool = True) -> list[Path]:
    """Salva a figura em PNG (alta resolução) e PDF.

    Args:
        fig: figura Matplotlib.
        pasta: diretório de destino (criado se necessário).
        nome: nome do arquivo sem extensão.
        formatos: extensões; por padrão as da configuração (png, pdf).
        fechar: fecha a figura após salvar (evita acúmulo de memória no pipeline).

    Returns:
        Lista de caminhos gravados.
    """
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for ext in formatos or _CFG["formatos"]:
        destino = pasta / f"{nome}.{ext}"
        fig.savefig(destino, dpi=_CFG["dpi"], bbox_inches="tight", facecolor="white")
        caminhos.append(destino)
    if fechar:
        plt.close(fig)
    return caminhos


def formatar_eixo_datas(ax: plt.Axes, curto: bool = False) -> None:
    """Datas no eixo x: 'mês/ano' em séries longas e 'dd/mm' em janelas curtas."""
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6 if curto else 8))
    if curto:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    else:
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda x, _: f"{MESES_PT[mdates.num2date(x).month - 1]}/{mdates.num2date(x):%y}"))


def anotar_quebras(ax: plt.Axes, quebras: dict[str, pd.Timestamp], y_rel: float = 0.97) -> None:
    """Linhas verticais tracejadas com rótulo para cada evento/quebra estrutural.

    Args:
        quebras: {rótulo: data}.
        y_rel: altura relativa (0–1) do rótulo dentro do eixo.
    """
    validas = [(r, d) for r, d in quebras.items() if d is not None]
    for i, (rotulo, data) in enumerate(validas):
        ax.axvline(data, color=CORES["quebra"], ls="--", lw=1.1, zorder=1)
        # rótulos alternam de altura para não colidirem quando as datas são próximas
        ax.annotate(rotulo, xy=(data, y_rel - 0.07 * (i % 2)), xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", fontsize=8.5, color=CORES["texto_secundario"], va="top",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def reservar_topo(ax: plt.Axes, fracao: float = 0.25) -> None:
    """Amplia o eixo y para deixar uma faixa livre no topo, onde ficam a legenda e os rótulos de eventos.

    Legendas e anotações são posicionadas em coordenadas do eixo; sem essa folga, elas ficam por cima
    dos dados quando a série chega perto do topo. Deve ser chamada depois de todos os dados plotados.

    Args:
        fracao: acréscimo ao intervalo do eixo y (0,25 = os dados ocupam ~80% da altura).
    """
    baixo, alto = ax.get_ylim()
    ax.set_ylim(baixo, alto + fracao * (alto - baixo))


def sombrear_periodo(ax: plt.Axes, inicio: pd.Timestamp, fim: pd.Timestamp, rotulo: str | None = "Teste") -> None:
    """Sombreia um intervalo de datas (ex.: período de teste)."""
    ax.axvspan(inicio - pd.Timedelta(hours=12), fim + pd.Timedelta(hours=12), color=CORES["fundo_teste"],
               zorder=0, label=rotulo)


def formatar_milhar(ax: plt.Axes, eixo: str = "y") -> None:
    """Separador de milhar no padrão brasileiro."""
    fmt = mtick.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(fmt)


def formatar_percentual(ax: plt.Axes, eixo: str = "y", escala: float = 1.0) -> None:
    """Eixo em percentual (``escala=1`` para frações 0–1, ``100`` para valores já em %)."""
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(mtick.PercentFormatter(escala, decimals=0))


def nota_rodape(fig: plt.Figure, texto: str) -> None:
    """Fonte/nota metodológica no rodapé da figura."""
    fig.text(0.01, -0.02, texto, fontsize=8, color=CORES["texto_secundario"], ha="left", va="top")


def plot_serie_com_media(ax: plt.Axes, serie: pd.Series, nome: str, unidade: str,
                         quebras: dict[str, pd.Timestamp] | None = None, janela: int = 7) -> None:
    """Série diária (cinza) com média móvel na cor da entidade e quebras anotadas."""
    ax.plot(serie.index, serie.values, color=CORES["contexto"], lw=0.8, label="Diário")
    ax.plot(serie.index, serie.rolling(janela, min_periods=janela).mean(), color=cor_serie(nome), lw=2.2,
            label=f"Média móvel de {janela} dias")
    ax.set_ylabel(unidade)
    formatar_eixo_datas(ax)
    if quebras:
        anotar_quebras(ax, quebras)
    ax.legend(loc="upper left")


def plot_previsao(ax: plt.Axes, historico: pd.Series, observado: pd.Series | None, previsao: pd.Series,
                  intervalos: pd.DataFrame | None, cor: str, titulo: str, unidade: str,
                  rotulo_previsao: str = "Previsto") -> None:
    """Histórico recente + observado no teste + previsão com intervalos de 80% e 95%."""
    ax.plot(historico.index, historico.values, color=CORES["contexto"], lw=1.2, label="Histórico")
    if intervalos is not None:
        ax.fill_between(previsao.index, intervalos["li_95"], intervalos["ls_95"], color=cor, alpha=0.13, lw=0,
                        label="Intervalo de 95%")
        ax.fill_between(previsao.index, intervalos["li_80"], intervalos["ls_80"], color=cor, alpha=0.28, lw=0,
                        label="Intervalo de 80%")
    if observado is not None:
        ax.plot(observado.index, observado.values, color=CORES["observado"], marker="o", ms=4.5, lw=1.4,
                label="Observado")
    ax.plot(previsao.index, previsao.values, color=cor, marker="o", ms=4.5, lw=2, ls="--", label=rotulo_previsao)
    ax.set_title(titulo)
    ax.set_ylabel(unidade)
    ax.set_ylim(bottom=0)
    reservar_topo(ax, 0.25)
    formatar_eixo_datas(ax, curto=True)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
