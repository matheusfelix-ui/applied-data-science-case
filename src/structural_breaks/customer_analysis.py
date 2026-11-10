"""Quebra final: o aumento de patamar de 2025 está associado à entrada de novos clientes?

**Proxy de cliente.** O identificador anonimizado ``card`` da Máquina 1. Limitações explícitas:
- um cartão não é uma pessoa (uma pessoa pode ter vários cartões; cartões podem ser reemitidos);
- vendas em dinheiro não têm ``card`` (na M1, 89 vendas, todas até jun/2024: depois de jul/2024 a
  M1 só registra cartão, o que torna a identificação **completa** no período da quebra);
- a Máquina 2 não tem coluna ``card``: a análise de clientes se restringe à M1;
- **censura à esquerda**: em mar/2024 todo cartão é "novo" por construção. Os primeiros
  ``burn_in_clientes_dias`` (28) dias não entram nas comparações de "novos".

**Perguntas e métodos.**
1. Evolução semanal de cartões únicos, novos (primeira compra na semana) e recorrentes.
2. Decomposição do crescimento: Δ demanda diária (depois − antes da quebra) em cartões novos
   (primeira compra após a quebra), recorrentes (já vistos antes) e sem identificação.
3. O afluxo de novos é anormal? Proporção semanal de transações de cartões "recém-chegados"
   (primeira compra nos 28 dias anteriores) depois da quebra × distribuição histórica.
4. Frequência de compra por cartão antes × depois (Mann-Whitney) e retenção dos novos.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.settings import carregar_config
from src.visualization.plots import CORES, anotar_quebras, cor_serie, formatar_eixo_datas

_CFG = carregar_config()["quebras"]


def preparar_cartoes(transacoes: pd.DataFrame) -> pd.DataFrame:
    """Transações da M1 com cartão, marcando a data da primeira compra de cada cartão."""
    m1 = transacoes[(transacoes["maquina"] == 1) & transacoes["card"].notna()].copy()
    m1["primeira_compra"] = m1.groupby("card")["date"].transform("min")
    return m1


def evolucao_semanal(transacoes: pd.DataFrame) -> pd.DataFrame:
    """Por semana: cartões únicos, novos, recorrentes e transações de cada grupo (M1)."""
    m1 = preparar_cartoes(transacoes)
    inicio_valido = transacoes["date"].min() + pd.Timedelta(days=_CFG["burn_in_clientes_dias"])
    m1["semana"] = m1["date"].dt.to_period("W-SUN").dt.start_time
    m1["novo"] = m1["date"].dt.to_period("W-SUN") == m1["primeira_compra"].dt.to_period("W-SUN")
    m1["recem_chegado"] = (m1["date"] - m1["primeira_compra"]).dt.days < 28
    sem = m1.groupby("semana").agg(cartoes_unicos=("card", "nunique"),
                                   transacoes=("card", "size"),
                                   transacoes_de_novos=("novo", "sum"),
                                   transacoes_recem_chegados=("recem_chegado", "sum"))
    sem["cartoes_novos"] = m1[m1["novo"]].groupby("semana")["card"].nunique().reindex(sem.index, fill_value=0)
    sem["cartoes_recorrentes"] = sem["cartoes_unicos"] - sem["cartoes_novos"]
    sem["% transações de recém-chegados"] = 100 * sem["transacoes_recem_chegados"] / sem["transacoes"]
    sem["compras por cartão"] = sem["transacoes"] / sem["cartoes_unicos"]
    sem["válida (fora do burn-in)"] = sem.index >= inicio_valido
    sem_cartao = transacoes[(transacoes["maquina"] == 1) & transacoes["card"].isna()]
    sem["transacoes_sem_cartao"] = sem_cartao.groupby(sem_cartao["date"].dt.to_period("W-SUN").dt.start_time).size() \
        .reindex(sem.index, fill_value=0)
    return sem


def decompor_crescimento(transacoes: pd.DataFrame, quebra: pd.Timestamp, fim: pd.Timestamp,
                         semanas_antes: int = 8) -> pd.DataFrame:
    """Decompõe o aumento da demanda diária da M1 após a quebra por tipo de cliente.

    Janela "antes": as ``semanas_antes`` semanas anteriores à quebra. "Depois": da quebra a ``fim``.
    Cartões "novos" = primeira compra a partir da quebra (nunca vistos no histórico anterior).
    """
    m1 = transacoes[transacoes["maquina"] == 1]
    inicio_antes = quebra - pd.Timedelta(weeks=semanas_antes)
    antes = m1[(m1["date"] >= inicio_antes) & (m1["date"] < quebra)]
    depois = m1[(m1["date"] >= quebra) & (m1["date"] <= fim)]
    vistos_antes = set(m1.loc[m1["date"] < quebra, "card"].dropna())
    dias_antes, dias_depois = (quebra - inicio_antes).days, (fim - quebra).days + 1

    def grupo(d, conjunto_novos):
        sem = d["card"].isna()
        novos = d["card"].notna() & ~d["card"].isin(vistos_antes) if conjunto_novos else pd.Series(False, index=d.index)
        rec = d["card"].notna() & ~novos
        return {"novos (1ª compra após a quebra)": novos.sum(), "recorrentes (já vistos antes)": rec.sum(), "sem cartão": sem.sum()}

    ga, gd = grupo(antes, False), grupo(depois, True)
    linhas = []
    total_delta = len(depois) / dias_depois - len(antes) / dias_antes
    for k in gd:
        taxa_a, taxa_d = ga[k] / dias_antes, gd[k] / dias_depois
        linhas.append({"grupo": k, "transações/dia antes": taxa_a, "transações/dia depois": taxa_d,
                       "Δ transações/dia": taxa_d - taxa_a, "% do crescimento": 100 * (taxa_d - taxa_a) / total_delta})
    tabela = pd.DataFrame(linhas).set_index("grupo")
    tabela.loc["total"] = [len(antes) / dias_antes, len(depois) / dias_depois, total_delta, 100.0]
    novos = depois.loc[depois["card"].notna() & ~depois["card"].isin(vistos_antes), "card"]
    tabela.attrs.update(cartoes_depois=depois["card"].nunique(), cartoes_novos=novos.nunique(),
                        janela_antes=f"{inicio_antes:%d/%m/%Y}–{quebra - pd.Timedelta(days=1):%d/%m/%Y}",
                        janela_depois=f"{quebra:%d/%m/%Y}–{fim:%d/%m/%Y}")
    return tabela


def teste_afluxo_novos(semanal: pd.DataFrame, quebra: pd.Timestamp) -> dict:
    """O % de transações de recém-chegados após a quebra é anormal frente ao histórico?

    Compara a média pós-quebra com a distribuição de médias de blocos móveis do mesmo tamanho
    no histórico anterior (fora do burn-in): p empírico = fração de blocos históricos ≥ observado.
    Complementa com Mann-Whitney semanal (supõe semanas independentes; leitura indicativa).
    """
    s = semanal[semanal["válida (fora do burn-in)"]]["% transações de recém-chegados"]
    depois = s[s.index >= quebra - pd.Timedelta(days=6)]
    antes = s[s.index < quebra - pd.Timedelta(days=6)]
    k = len(depois)
    blocos = antes.rolling(k).mean().dropna()
    return {"semanas depois": k, "média depois (%)": depois.mean(), "média histórica (%)": antes.mean(),
            "máximo histórico de blocos de mesmo tamanho (%)": blocos.max(),
            "p empírico (blocos históricos ≥ observado)": float((blocos >= depois.mean()).mean()) if len(blocos) else np.nan,
            "Mann-Whitney p (semanas)": stats.mannwhitneyu(depois, antes, alternative="greater").pvalue}


def frequencia_e_retencao(transacoes: pd.DataFrame, quebra: pd.Timestamp, fim: pd.Timestamp,
                          semanas: int = 6) -> pd.DataFrame:
    """Compras por cartão e % de cartões que voltam, para coortes de novos de mesma duração.

    Coorte = cartões cuja primeira compra cai numa janela de ``semanas`` semanas; acompanhamento
    pelas mesmas ``semanas`` semanas após o início da coorte (comparável entre coortes).
    """
    m1 = preparar_cartoes(transacoes)
    inicio_valido = transacoes["date"].min() + pd.Timedelta(days=_CFG["burn_in_clientes_dias"])
    duracao = pd.Timedelta(weeks=semanas)
    inicios = list(pd.date_range(inicio_valido, quebra - duracao, freq=duracao)) + [quebra]
    linhas = []
    for ini in inicios:
        coorte = m1[(m1["primeira_compra"] >= ini) & (m1["primeira_compra"] < ini + duracao)]
        janela = coorte[coorte["date"] < min(ini + duracao, fim + pd.Timedelta(days=1))]
        if janela.empty:
            continue
        por_cartao = janela.groupby("card").size()
        linhas.append({"coorte (1ª compra a partir de)": ini, "pós-quebra": ini >= quebra, "cartões novos": len(por_cartao),
                       "novos por semana": len(por_cartao) / semanas,
                       "compras por cartão (na janela)": por_cartao.mean(),
                       "% que compram 2+ vezes": 100 * (por_cartao >= 2).mean()})
    return pd.DataFrame(linhas)


def comparar_frequencia(transacoes: pd.DataFrame, quebra: pd.Timestamp, fim: pd.Timestamp, semanas: int = 6) -> dict:
    """Compras por cartão ativo em janelas de mesma duração antes e depois da quebra (Mann-Whitney)."""
    m1 = preparar_cartoes(transacoes)
    dur = pd.Timedelta(weeks=semanas)
    a = m1[(m1["date"] >= quebra - dur) & (m1["date"] < quebra)].groupby("card").size()
    d = m1[(m1["date"] >= quebra) & (m1["date"] < min(quebra + dur, fim + pd.Timedelta(days=1)))].groupby("card").size()
    return {"cartões ativos antes": len(a), "cartões ativos depois": len(d),
            "compras/cartão antes (média)": a.mean(), "compras/cartão depois (média)": d.mean(),
            "Mann-Whitney p": stats.mannwhitneyu(a, d).pvalue}


def grafico_clientes(semanal: pd.DataFrame, decomposicao: pd.DataFrame, quebra: pd.Timestamp) -> plt.Figure:
    """Cartões novos × recorrentes por semana e decomposição do crescimento pós-quebra."""
    fig, eixos = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = eixos[0]
    s = semanal[semanal["válida (fora do burn-in)"]]
    ax.bar(s.index, s["cartoes_recorrentes"], width=5, color=CORES["contexto"], label="Cartões recorrentes",
           edgecolor="white", lw=1)
    ax.bar(s.index, s["cartoes_novos"], width=5, bottom=s["cartoes_recorrentes"], color=cor_serie("Máquina 1"),
           label="Cartões novos (1ª compra na semana)", edgecolor="white", lw=1)
    anotar_quebras(ax, {f"quebra {quebra:%d/%m/%y}": quebra})
    ax.set_ylabel("cartões únicos na semana")
    ax.set_title("Máquina 1: cartões ativos por semana (após 28 dias de burn-in)")
    formatar_eixo_datas(ax)
    ax.legend(loc="upper left")
    ax = eixos[1]
    d = decomposicao.drop(index="total")
    cores = [cor_serie("Máquina 1"), CORES["contexto"], CORES["observado"]]
    ax.barh(d.index, d["Δ transações/dia"], color=cores, height=0.55)
    for k, (nome, r) in enumerate(d.iterrows()):
        ax.annotate(f"{r['Δ transações/dia']:+.1f}/dia ({r['% do crescimento']:.0f}%)", (max(r["Δ transações/dia"], 0), k),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    ax.axvline(0, color=CORES["eixo"], lw=1)
    ax.invert_yaxis()
    ax.set_xlabel("Δ transações/dia (depois − antes)")
    ax.set_title(f"De onde veio o aumento de {decomposicao.loc['total', 'Δ transações/dia']:+.1f}/dia")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.6)
    fig.tight_layout()
    return fig
