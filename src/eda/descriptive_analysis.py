"""Análise descritiva: distribuição, evolução temporal e padrões por horário e dia da semana.

As médias por hora e dia da semana usam **todos os dias do calendário** do período ativo no
denominador (inclusive dias sem venda), para não superestimar a demanda típica.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.data.preprocessing import BaseDados
from src.visualization.plots import (CORES, DIAS_PT, MESES_PT, SEQUENCIAL, cor_serie, formatar_eixo_datas,
                                     anotar_quebras, plot_serie_com_media)

UNIDADES = {"demanda": "transações/dia", "receita": "receita/dia (u.m.)"}


def resumo_series(base: BaseDados) -> pd.DataFrame:
    """Estatísticas das séries diárias (demanda e receita) e do ticket médio por série."""
    linhas = []
    for (serie, alvo), s in base.series.items():
        linhas.append({
            "série": serie, "alvo": alvo, "início": f"{s.index.min():%d/%m/%Y}", "dias": len(s),
            "total": s.sum(), "média": s.mean(), "mediana": s.median(), "desvio-padrão": s.std(),
            "CV": s.std() / s.mean(), "mínimo": s.min(), "p5": s.quantile(0.05), "p95": s.quantile(0.95),
            "máximo": s.max(), "assimetria": stats.skew(s), "var/média": s.var() / s.mean() if alvo == "demanda" else np.nan,
        })
    tabela = pd.DataFrame(linhas)
    ticket = {m: base.series[(m, "receita")].sum() / base.series[(m, "demanda")].sum() for m in base.inicio}
    tabela["ticket médio"] = tabela["série"].map(ticket)
    return tabela


def comparar_periodo_comum(base: BaseDados) -> pd.DataFrame:
    """Máquina 1 × Máquina 2 no período em que as duas coexistem (08/02 a 23/03/2025)."""
    inicio = base.inicio["Máquina 2"]
    linhas = []
    for m in ("Máquina 1", "Máquina 2"):
        d = base.series[(m, "demanda")].loc[inicio:]
        r = base.series[(m, "receita")].loc[inicio:]
        linhas.append({"série": m, "dias": len(d), "demanda média": d.mean(), "CV demanda": d.std() / d.mean(),
                       "receita média": r.mean(), "ticket médio": r.sum() / d.sum(),
                       "% demanda no fim de semana": 100 * d[d.index.dayofweek >= 5].sum() / d.sum(),
                       "dias sem venda": int((d == 0).sum())})
    tabela = pd.DataFrame(linhas).set_index("série")
    d1, d2 = (base.series[(m, "demanda")].loc[inicio:] for m in ("Máquina 1", "Máquina 2"))
    tabela.attrs["mann_whitney_p"] = stats.mannwhitneyu(d1, d2).pvalue
    tabela.attrs["spearman"] = stats.spearmanr(d1, d2)
    return tabela


def media_hora_dia(transacoes: pd.DataFrame, inicio: pd.Timestamp, fim: pd.Timestamp) -> pd.DataFrame:
    """Média de transações por hora × dia da semana (zeros explícitos para dias/horas sem venda)."""
    calendario = pd.date_range(inicio, fim, freq="D")
    contagem = (transacoes.groupby(["date", "hora"]).size().unstack(fill_value=0)
                .reindex(index=calendario, columns=range(6, 24), fill_value=0))
    return contagem.groupby(calendario.dayofweek).mean().rename(index=dict(enumerate(DIAS_PT))).T


def media_mensal(base: BaseDados) -> pd.DataFrame:
    """Demanda e receita médias por dia em cada mês (ajusta meses incompletos)."""
    linhas = []
    for (serie, alvo), s in base.series.items():
        m = s.groupby(s.index.to_period("M")).mean()
        linhas.append(m.rename(f"{serie} — {alvo}"))
    return pd.concat(linhas, axis=1)


# ------------------------------------------------------------------------------------- gráficos
def grafico_distribuicoes(base: BaseDados) -> plt.Figure:
    """Histogramas da demanda e da receita diárias por série (período ativo de cada uma)."""
    fig, eixos = plt.subplots(2, 3, figsize=(16, 7.5))
    for i, alvo in enumerate(("demanda", "receita")):
        for j, serie in enumerate(("Máquina 1", "Máquina 2", "Total")):
            ax, s = eixos[i, j], base.series[(serie, alvo)]
            discreto = alvo == "demanda"
            sns.histplot(s, ax=ax, color=cor_serie(serie), discrete=discreto, bins=None if discreto else 25,
                         edgecolor="white", linewidth=1.2)
            ax.axvline(s.mean(), color=CORES["observado"], ls="--", lw=1.2, label=f"média = {s.mean():.1f}")
            ax.axvline(s.median(), color=CORES["texto_secundario"], ls=":", lw=1.2, label=f"mediana = {s.median():.1f}")
            ax.set_title(f"{serie} — {alvo}")
            ax.set_xlabel(UNIDADES[alvo])
            ax.set_ylabel("nº de dias")
            ax.legend(loc="upper right")
    fig.suptitle("Distribuição diária da demanda e da receita (assimetria à direita; a M2 tem só 44 dias)")
    fig.tight_layout()
    return fig


def grafico_evolucao(base: BaseDados, quebras: dict[str, pd.Timestamp] | None = None) -> plt.Figure:
    """Séries diárias com média móvel de 7 dias e eventos anotados (3 séries × 2 alvos)."""
    eventos = {"Entrada da M2": base.inicio["Máquina 2"], **(quebras or {})}
    fig, eixos = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    indice = base.series[("Total", "demanda")].index
    for j, alvo in enumerate(("demanda", "receita")):
        for i, serie in enumerate(("Máquina 1", "Máquina 2", "Total")):
            ax = eixos[i, j]
            s = base.series[(serie, alvo)].reindex(indice)
            plot_serie_com_media(ax, s, serie, UNIDADES[alvo], eventos if serie != "Máquina 2" else None)
            ax.set_title(f"{serie} — {alvo}")
            if serie == "Máquina 2":
                ax.annotate("Máquina 2 só registra vendas a partir de 08/02/2025", xy=(0.02, 0.5),
                            xycoords="axes fraction", fontsize=9, color=CORES["texto_secundario"])
    fig.suptitle("Evolução diária da demanda e da receita, por série")
    fig.tight_layout()
    return fig


def grafico_hora_dia(base: BaseDados) -> plt.Figure:
    """Heatmaps de média de transações por hora × dia da semana, por máquina."""
    fig, eixos = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, serie in zip(eixos, ("Máquina 1", "Máquina 2")):
        t = base.transacoes[base.transacoes["serie"] == serie]
        tabela = media_hora_dia(t, base.inicio[serie], base.fim)
        sns.heatmap(tabela, ax=ax, cmap=SEQUENCIAL, annot=True, fmt=".2f", annot_kws={"size": 7.5},
                    linewidths=0.6, linecolor="white", cbar_kws={"label": "transações médias por hora"})
        ax.set_title(f"{serie} ({base.inicio[serie]:%d/%m/%Y} a {base.fim:%d/%m/%Y})")
        ax.set_xlabel("")
        ax.set_ylabel("Hora do dia")
        ax.set_yticklabels([f"{h:02d}h" for h in range(6, 24)], rotation=0)
    fig.suptitle("Demanda média por hora e dia da semana (dias sem venda incluídos no denominador)")
    fig.tight_layout()
    return fig


def grafico_perfis(base: BaseDados) -> plt.Figure:
    """Perfil semanal e horário relativos (índice = média do dia ou hora ÷ média geral)."""
    fig, eixos = plt.subplots(1, 2, figsize=(15, 4.5))
    inicio = base.inicio["Máquina 2"]
    for serie, periodo, estilo in [("Máquina 1", "todo o período", "-"), ("Máquina 1", "período comum", ":"),
                                   ("Máquina 2", "período comum", "-")]:
        s = base.series[(serie, "demanda")]
        s = s.loc[inicio:] if periodo == "período comum" else s
        perfil = s.groupby(s.index.dayofweek).mean() / s.mean()
        eixos[0].plot(DIAS_PT, perfil.values, marker="o", ms=7, ls=estilo, color=cor_serie(serie),
                      label=f"{serie} ({periodo})")
    eixos[0].axhline(1, color=CORES["contexto"], lw=1)
    eixos[0].set_title("Perfil semanal relativo da demanda")
    eixos[0].set_ylabel("índice (média do dia ÷ média geral)")
    eixos[0].legend(loc="upper right")
    for serie in ("Máquina 1", "Máquina 2"):
        t = base.transacoes[(base.transacoes["serie"] == serie) & (base.transacoes["date"] >= inicio)]
        perfil = t["hora"].value_counts(normalize=True).reindex(range(6, 24), fill_value=0)
        eixos[1].plot(perfil.index, perfil.values, marker="o", ms=6, color=cor_serie(serie), label=f"{serie} (período comum)")
    eixos[1].set_xticks(range(6, 24, 2))
    eixos[1].set_xticklabels([f"{h:02d}h" for h in range(6, 24, 2)])
    eixos[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    eixos[1].set_title("Distribuição das vendas por hora")
    eixos[1].set_ylabel("% das transações")
    eixos[1].legend(loc="upper right")
    fig.tight_layout()
    return fig


def grafico_mensal(base: BaseDados) -> plt.Figure:
    """Demanda média diária por mês, por série (barras agrupadas)."""
    tabela = media_mensal(base)
    fig, ax = plt.subplots(figsize=(15, 4.6))
    meses = tabela.index
    x = np.arange(len(meses))
    largura = 0.28
    for k, serie in enumerate(("Máquina 1", "Máquina 2", "Total")):
        valores = tabela[f"{serie} — demanda"].reindex(meses)
        ax.bar(x + (k - 1) * largura, valores.values, width=largura - 0.03, color=cor_serie(serie), label=serie)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MESES_PT[p.month - 1]}/{p.year % 100}" for p in meses])
    ax.set_ylabel("transações/dia (média do mês)")
    ax.set_title("Demanda média diária por mês (mar/2024 e mar/2025 incompletos; a M2 começa em fev/2025)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
