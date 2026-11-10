"""Produtos e mix de vendas: participação, concentração (HHI), preços e evolução do mix."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.preprocessing import BaseDados
from src.visualization.plots import CORES, MESES_PT, cor_serie, formatar_eixo_datas

# Paleta categórica de 8 posições, em ordem fixa (validada para daltonismo em pares adjacentes)
PALETA_PRODUTOS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def mix_produtos(base: BaseDados, serie: str) -> pd.DataFrame:
    """Transações, receita, participação e preço médio por produto em uma série."""
    t = base.transacoes if serie == "Total" else base.transacoes[base.transacoes["serie"] == serie]
    tabela = t.groupby("produto").agg(transacoes=("money", "size"), receita=("money", "sum"),
                                      preco_medio=("money", "mean"), precos_distintos=("money", "nunique"))
    tabela["% transações"] = 100 * tabela["transacoes"] / tabela["transacoes"].sum()
    tabela["% receita"] = 100 * tabela["receita"] / tabela["receita"].sum()
    return tabela.sort_values("receita", ascending=False)


def hhi(participacoes_pct: pd.Series) -> float:
    """Índice Herfindahl-Hirschman (0–10.000) a partir de participações em %."""
    return float((participacoes_pct ** 2).sum())


def concentracao(base: BaseDados) -> pd.DataFrame:
    """HHI da receita, nº de produtos e participação dos 3 maiores, por série."""
    linhas = []
    for serie in ("Máquina 1", "Máquina 2", "Total"):
        mix = mix_produtos(base, serie)
        linhas.append({"série": serie, "produtos": len(mix), "HHI da receita": hhi(mix["% receita"]),
                       "HHI equivalente a N produtos iguais": 10_000 / hhi(mix["% receita"]),
                       "% receita dos 3 maiores": mix["% receita"].head(3).sum()})
    return pd.DataFrame(linhas).set_index("série")


def precos_mensais(base: BaseDados, serie: str = "Máquina 1") -> pd.DataFrame:
    """Preço modal por produto e mês (evidencia as mudanças de tabela de preços)."""
    t = base.transacoes[base.transacoes["serie"] == serie]
    return t.groupby([t["date"].dt.to_period("M"), "produto"])["money"].agg(lambda s: s.mode()[0]).unstack()


def mudancas_preco(base: BaseDados, serie: str = "Máquina 1", min_dias: int = 7, min_variacao: float = 2.0) -> pd.DataFrame:
    """Mudanças persistentes de tabela de preço (só vendas em cartão).

    Até jun/2024, o preço em dinheiro era arredondado (ex.: 40,00 × 38,70 no cartão), o que gera
    falsas "mudanças" diárias. Por isso usamos apenas cartão e só contamos uma mudança se o novo
    preço vale por pelo menos ``min_dias`` dias com venda e varia mais de ``min_variacao`` %.
    """
    t = base.transacoes[(base.transacoes["serie"] == serie) & (base.transacoes["cash_type"] == "card")]
    linhas = []
    for produto, g in t.sort_values("datetime").groupby("produto"):
        diario = g.groupby("date")["money"].agg(lambda s: s.mode()[0]).round(2)
        blocos = (diario != diario.shift()).cumsum()
        segmentos = diario.groupby(blocos).agg(["first", "size"])
        segmentos["inicio"] = diario.index.to_series().groupby(blocos.values).first().values
        segmentos = segmentos[segmentos["size"] >= min_dias].reset_index(drop=True)
        for k in range(1, len(segmentos)):
            ant, novo = segmentos.loc[k - 1, "first"], segmentos.loc[k, "first"]
            if abs(novo / ant - 1) * 100 > min_variacao:
                linhas.append({"produto": produto, "data": segmentos.loc[k, "inicio"], "preço anterior": ant,
                               "preço novo": novo, "variação %": 100 * (novo / ant - 1)})
    return pd.DataFrame(linhas).sort_values("data").reset_index(drop=True)


def grafico_mix(base: BaseDados) -> plt.Figure:
    """Participação na receita dos produtos (top 8 + 'Outros') por máquina."""
    fig, eixos = plt.subplots(1, 2, figsize=(15, 5))
    for ax, serie in zip(eixos, ("Máquina 1", "Máquina 2")):
        mix = mix_produtos(base, serie)["% receita"]
        top = mix.head(8)
        if len(mix) > 8:
            top = pd.concat([top, pd.Series({"Outros": mix.iloc[8:].sum()})])
        top = top[::-1]
        ax.barh(top.index, top.values, color=cor_serie(serie), height=0.65)
        for k, (nome, v) in enumerate(top.items()):
            ax.annotate(f"{v:.0f}%", (v, k), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8.5,
                        color=CORES["texto_secundario"])
        ax.set_xlabel("% da receita da máquina")
        ax.set_title(f"{serie}: {len(mix)} produtos")
    fig.suptitle("Mix de produtos por receita (a M2 tem cardápio muito mais amplo)")
    fig.tight_layout()
    return fig


def grafico_evolucao_mix(base: BaseDados) -> plt.Figure:
    """Máquina 1: participação mensal dos produtos (barras 100% empilhadas) e preço modal por produto."""
    t = base.transacoes[base.transacoes["serie"] == "Máquina 1"]
    mes = t["date"].dt.to_period("M")
    part = pd.crosstab(mes, t["produto"], normalize="index") * 100
    ordem = part.mean().sort_values(ascending=False).index
    part = part[ordem]
    fig, eixos = plt.subplots(1, 2, figsize=(16, 5.2))
    ax = eixos[0]
    base_y = np.zeros(len(part))
    x = np.arange(len(part))
    for k, produto in enumerate(part.columns):
        ax.bar(x, part[produto], bottom=base_y, color=PALETA_PRODUTOS[k % 8], width=0.8, label=produto,
               edgecolor="white", linewidth=1.5)
        base_y += part[produto].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MESES_PT[p.month - 1]}/{p.year % 100}" for p in part.index], rotation=45)
    ax.set_ylabel("% das transações do mês")
    ax.set_title("Máquina 1: mix mensal de produtos")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    ax = eixos[1]
    precos = t.groupby(["date", "produto"])["money"].median().unstack()[ordem]
    for k, produto in enumerate(precos.columns):
        ax.step(precos.index, precos[produto].ffill(), where="post", color=PALETA_PRODUTOS[k % 8], lw=1.8, label=produto)
    ax.set_ylabel("preço (u.m.)")
    ax.set_title("Máquina 1: preço por produto (tabelas mudam em jul/2024 e out/2024)")
    formatar_eixo_datas(ax)
    fig.tight_layout()
    return fig


def grafico_ticket(base: BaseDados) -> plt.Figure:
    """Ticket médio semanal por máquina (receita ÷ transações)."""
    fig, ax = plt.subplots(figsize=(15, 4))
    for serie, ticket in ticket_semanal(base).items():
        ax.plot(ticket.index, ticket.values, marker="o", ms=4, color=cor_serie(serie), label=serie)
    ax.set_ylabel("ticket médio semanal (u.m.)")
    ax.set_title("Ticket médio: a receita da M1 muda com a tabela de preços, não só com o volume")
    formatar_eixo_datas(ax)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


# ============================================================================ visão de negócio: concentração, cauda e público
# Famílias de insumo inferidas pelo nome do produto (a base não traz receitas/fichas técnicas).
FAMILIAS_INSUMO = {"leite": "milk|latte|cappuccino|cortado|mochaccino", "chocolate/cacau": "chocolate|cocoa",
                   "whiskey (aroma)": "whiskey|irish", "caramelo": "caramel", "baunilha": "vanilla", "chá": r"\btea\b"}


def curva_pareto(base: BaseDados, serie: str) -> pd.DataFrame:
    """Produtos em ordem de receita com participação acumulada (curva ABC)."""
    mix = mix_produtos(base, serie)[["transacoes", "receita", "% receita"]].copy()
    mix["% acumulado"] = mix["% receita"].cumsum()
    mix["posição"] = np.arange(1, len(mix) + 1)
    mix["classe"] = np.where(mix["% acumulado"].shift(fill_value=0) < 80, "A",
                             np.where(mix["% acumulado"].shift(fill_value=0) < 95, "B", "C"))
    return mix


def resumo_concentracao_negocio(base: BaseDados, max_vendas_cauda: int = 5) -> pd.DataFrame:
    """Quantos produtos fazem 50%/80% da receita e quanto pesa a cauda de baixo giro, por série.

    Cauda = produtos com até ``max_vendas_cauda`` vendas no período ativo da série
    (para a M2, 44 dias: menos de uma venda por semana).
    """
    linhas = []
    for serie in ("Máquina 1", "Máquina 2", "Total"):
        p = curva_pareto(base, serie)
        semanas = base.series[(serie, "demanda")].shape[0] / 7
        cauda = p[p["transacoes"] <= max_vendas_cauda]
        linhas.append({"série": serie, "produtos": len(p), "produtos para 50% da receita": int((p["% acumulado"] < 50).sum() + 1),
                       "produtos para 80% da receita": int((p["% acumulado"] < 80).sum() + 1),
                       "% receita dos 3 maiores": p["% receita"].head(3).sum(),
                       f"produtos com ≤ {max_vendas_cauda} vendas": len(cauda), "% receita da cauda": cauda["% receita"].sum(),
                       "produtos vendidos < 1×/semana": int((p["transacoes"] / semanas < 1).sum())})
    return pd.DataFrame(linhas).set_index("série")


def familias_insumo(base: BaseDados) -> pd.DataFrame:
    """Participação de cada família de insumo nas vendas e na receita de cada máquina.

    Mostra que insumos sairiam de linha se a cauda fosse cortada: uma família só deixa de ser
    estocada se **todos** os seus produtos saírem.
    """
    linhas = []
    for serie in ("Máquina 1", "Máquina 2"):
        t = base.transacoes[base.transacoes["serie"] == serie]
        nomes = t["produto"].str.lower()
        for fam, padrao in FAMILIAS_INSUMO.items():
            msk = nomes.str.contains(padrao, regex=True)
            linhas.append({"série": serie, "família": fam, "produtos": t.loc[msk, "produto"].nunique(), "vendas": int(msk.sum()),
                           "% vendas": 100 * msk.mean(), "% receita": 100 * t.loc[msk, "money"].sum() / t["money"].sum(),
                           "vendas/semana": msk.sum() / (base.series[(serie, "demanda")].shape[0] / 7)})
    return pd.DataFrame(linhas)


def ticket_semanal(base: BaseDados) -> pd.DataFrame:
    """Ticket médio semanal (receita ÷ transações) por máquina; semanas sem venda ficam NaN."""
    colunas = {}
    for serie in ("Máquina 1", "Máquina 2"):
        d = base.diarios[serie][["demanda", "receita"]].resample("W").sum()
        colunas[serie] = d["receita"] / d["demanda"].replace(0, np.nan)
    return pd.DataFrame(colunas)


def ticket_por_publico(base: BaseDados, quebra: pd.Timestamp, hora_noite: int = 19) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """O público mais recente gasta menos por compra? Separa efeito de preço e efeito de mix.

    Na M1, a última mudança de tabela de preços é anterior à ``quebra`` de 2025. Comparando o
    período entre a última mudança e a quebra com o período depois da quebra, os preços são os
    mesmos e toda a variação do ticket vem do **mix** (o que o público escolhe). A M2 entra como
    comparação no mesmo período (tabela própria, mais barata).

    Returns:
        (ticket por período, mix da M1 antes × depois com preço, resumo).
    """
    t = base.transacoes
    m1, m2 = t[t["serie"] == "Máquina 1"], t[t["serie"] == "Máquina 2"]
    ultima_tabela = mudancas_preco(base)["data"].max()
    antes = m1[(m1["date"] >= ultima_tabela) & (m1["date"] < quebra)]
    depois = m1[m1["date"] >= quebra]
    primeira_compra = m1.dropna(subset=["card"]).groupby("card")["date"].min()
    novos = depois[depois["card"].map(primeira_compra) >= quebra]
    antigos = depois[depois["card"].map(primeira_compra) < quebra]
    blocos = [("M1 · base anterior (mesma tabela)", antes), ("M1 · desde a quebra (todos)", depois),
              ("M1 · desde a quebra, cartões novos", novos), ("M1 · desde a quebra, cartões antigos", antigos),
              ("M2 · desde a abertura", m2)]
    tickets = pd.DataFrame([{"grupo": g, "início": d["date"].min(), "fim": d["date"].max(), "transações": len(d),
                             "ticket médio": d["money"].mean(), f"% vendas a partir das {hora_noite}h": 100 * (d["hora"] >= hora_noite).mean()}
                            for g, d in blocos]).set_index("grupo")
    mix = pd.concat([antes["produto"].value_counts(normalize=True).rename("% antes"),
                     depois["produto"].value_counts(normalize=True).rename("% depois")], axis=1).fillna(0) * 100
    mix["preço (u.m.)"] = m1[m1["date"] >= ultima_tabela].groupby("produto")["money"].median()
    mix["Δ p.p."] = mix["% depois"] - mix["% antes"]
    mix = mix.sort_values("Δ p.p.")
    # Mesmo produto, preço da M2 vs M1 no período comum
    precos = pd.concat([depois.groupby("produto")["money"].median().rename("M1"), m2.groupby("produto")["money"].median().rename("M2")], axis=1).dropna()
    resumo = {"última mudança de tabela da M1": ultima_tabela, "quebra": quebra,
              "ticket antes": antes["money"].mean(), "ticket depois": depois["money"].mean(),
              "variação do ticket da M1 (%)": 100 * (depois["money"].mean() / antes["money"].mean() - 1),
              "ticket M2": m2["money"].mean(), "M2 vs M1 no período comum (%)": 100 * (m2["money"].mean() / m1[m1["date"] >= m2["date"].min()]["money"].mean() - 1),
              "desconto da M2 nos mesmos produtos (%)": (100 * (precos["M2"] / precos["M1"] - 1)).round(1).to_dict(),
              "% dinheiro na M2": 100 * (m2["cash_type"] == "cash").mean()}
    return tickets, mix, resumo


def grafico_pareto(base: BaseDados) -> plt.Figure:
    """Curva de concentração da receita: % acumulado × nº de produtos, por máquina."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for serie in ("Máquina 1", "Máquina 2"):
        p = curva_pareto(base, serie)
        ax.plot(np.r_[0, p["posição"]], np.r_[0, p["% acumulado"]], marker="o", ms=4, color=cor_serie(serie),
                label=f"{serie} ({len(p)} produtos)")
    ax.axhline(80, color=CORES["contexto"], lw=1, ls="--")
    ax.annotate("80% da receita", (30, 80), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=8.5,
                color=CORES["texto_secundario"])
    ax.set_xlabel("nº de produtos (do maior para o menor em receita)")
    ax.set_ylabel("% acumulado da receita")
    ax.set_ylim(0, 102)
    ax.set_title("Concentração: poucos produtos fazem quase toda a receita")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
