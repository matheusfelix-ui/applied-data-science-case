"""Valores ausentes: quantificação, padrões e impacto.

Há três tipos de ausência, e cada uma tem tratamento próprio:

1. **Ausência de campo por meio de pagamento**: ``card`` é nulo nas vendas em dinheiro da Máquina 1
   (o identificador só existe para cartão). É informativa, não aleatória (MNAR por construção).
2. **Ausência estrutural de coluna**: ``index_2.csv`` não tem a coluna ``card``. Nada a imputar;
   limita a análise de clientes à Máquina 1.
3. **Ausência de período**: a Máquina 2 não existe antes de 08/02/2025. Não é demanda zero nem
   dado faltante a imputar; a série simplesmente começa nessa data.

As colunas usadas na previsão (``datetime`` e ``money``) não têm nenhum valor ausente.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.data.preprocessing import BaseDados
from src.visualization.plots import CORES, MESES_PT, cor_serie


def tabela_ausentes(base: BaseDados) -> pd.DataFrame:
    """Nulos por coluna e arquivo, distinguindo ausência de valor de ausência de coluna."""
    t = base.transacoes
    colunas = ["date", "datetime", "cash_type", "card", "money", "coffee_name"]
    linhas = []
    for arquivo, g in t.groupby("arquivo_origem"):
        for c in colunas:
            estrutural = c == "card" and g["card"].isna().all()
            linhas.append({"arquivo": arquivo, "coluna": c, "registros": len(g), "nulos": int(g[c].isna().sum()),
                           "% nulos": 100 * g[c].isna().mean(),
                           "tipo": "coluna inexistente no arquivo" if estrutural else
                                   ("sem ausência" if g[c].notna().all() else "valor ausente")})
    return pd.DataFrame(linhas)


def padrao_card_pagamento(base: BaseDados) -> pd.DataFrame:
    """Cruzamento ``card`` nulo × meio de pagamento na Máquina 1 (testa se a ausência é só do dinheiro)."""
    m1 = base.transacoes[base.transacoes["maquina"] == 1]
    return pd.crosstab(m1["cash_type"], m1["card"].isna().map({True: "card nulo", False: "card preenchido"}))


def ausencia_por_mes(base: BaseDados) -> pd.DataFrame:
    """Proporção mensal de ``card`` nulo e de vendas em dinheiro na Máquina 1."""
    m1 = base.transacoes[base.transacoes["maquina"] == 1]
    mes = m1["date"].dt.to_period("M")
    return pd.DataFrame({"% card nulo": m1["card"].isna().groupby(mes).mean() * 100,
                         "% em dinheiro": (m1["cash_type"] == "cash").groupby(mes).mean() * 100,
                         "transações": m1.groupby(mes).size()})


def cobertura_calendario(base: BaseDados) -> pd.DataFrame:
    """Dias do calendário global por série: com venda, sem venda (zero) e fora do período ativo."""
    calendario = base.series[("Total", "demanda")].index
    linhas = []
    for serie in ("Máquina 1", "Máquina 2", "Total"):
        s = base.series[(serie, "demanda")].reindex(calendario)
        linhas.append({"série": serie, "dias no calendário global": len(calendario),
                       "fora do período ativo (máquina inexistente)": int(s.isna().sum()),
                       "dias com venda": int((s > 0).sum()), "dias com zero vendas": int((s == 0).sum())})
    return pd.DataFrame(linhas).set_index("série")


def grafico_ausentes(base: BaseDados) -> plt.Figure:
    """Painel: % de ``card`` nulo × % em dinheiro por mês (M1) e cobertura do calendário por série."""
    mensal = ausencia_por_mes(base)
    fig, eixos = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = eixos[0]
    x = range(len(mensal))
    ax.bar(x, mensal["% em dinheiro"], color=cor_serie("Máquina 1"), width=0.6, label="% vendas em dinheiro")
    ax.plot(x, mensal["% card nulo"], color=CORES["observado"], marker="o", ms=6, lw=1.6, label="% com card nulo")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{MESES_PT[p.month - 1]}/{p.year % 100}" for p in mensal.index], rotation=0)
    ax.set_ylabel("% das transações do mês")
    ax.set_title("Máquina 1: card nulo coincide com venda em dinheiro (zero desde jul/2024)")
    ax.legend(loc="upper right")
    ax = eixos[1]
    cob = cobertura_calendario(base)
    esquerda = pd.Series(0, index=cob.index, dtype=float)
    for coluna, cor in [("dias com venda", CORES["Máquina 1"]), ("dias com zero vendas", CORES["observado"]),
                        ("fora do período ativo (máquina inexistente)", CORES["contexto"])]:
        ax.barh(cob.index, cob[coluna], left=esquerda, color=cor, label=coluna, height=0.55, edgecolor="white", lw=2)
        esquerda += cob[coluna]
    ax.invert_yaxis()
    ax.set_xlabel("dias (01/03/2024 a 23/03/2025)")
    ax.set_title("Cobertura do calendário por série")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=2, fontsize=8)
    fig.tight_layout()
    return fig
