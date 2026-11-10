"""Registros duplicados: erro de registro ou compra de mais de uma unidade?

A base não tem número de pedido nem log de entrega da máquina. Um registro repetido pode ser:

- **erro de registro** (a mesma venda gravada duas vezes): os dois registros ficariam a ~0 s um
  do outro, porque vêm do mesmo evento;
- **compra de duas unidades**: na M2 (caixa/PDV), os itens de um pedido saem no mesmo segundo; na
  M1 (máquina automática), a segunda bebida só sai depois de um ciclo completo de preparo.

Três verificações separam as duas hipóteses:

| Verificação | O que indicaria erro | O que indicaria compra |
|---|---|---|
| Duplicatas exatas (horário, produto, valor, pagamento) | aparecem isoladas | aparecem dentro de pedidos com vários itens |
| Pedidos no mesmo segundo (M2) | só produtos iguais | também produtos diferentes no mesmo segundo |
| Recompra rápida do mesmo produto pelo mesmo cartão (M1) | intervalos perto de 0 s | intervalo de uma nova preparação (~1 min), parecido com o das recompras de produtos diferentes |

Nada é removido: sem ID de pedido não dá para provar a causa, e apagar compras reais
subestimaria a demanda.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data.preprocessing import BaseDados
from src.visualization.plots import CORES, cor_serie

CHAVE_EXATA = ["datetime", "produto", "money", "cash_type"]


def duplicatas_exatas(base: BaseDados) -> pd.DataFrame:
    """Todos os registros envolvidos em duplicatas exatas (mesmo horário, produto, valor e pagamento)."""
    t = base.transacoes
    marc = t.groupby("serie", group_keys=False).apply(lambda g: g.duplicated(CHAVE_EXATA, keep=False), include_groups=False)
    return t.loc[marc[marc].index, ["serie", "datetime", "produto", "money", "cash_type", "card"]].sort_values(["serie", "datetime"])


def pedidos_mesmo_segundo(base: BaseDados) -> pd.DataFrame:
    """Grupos de registros com o mesmo segundo exato, por série: quantos itens e se os produtos são iguais."""
    t = base.transacoes.assign(segundo=base.transacoes["datetime"].dt.floor("s"))
    rep = t[t.duplicated(["serie", "segundo"], keep=False)]
    if rep.empty:
        return pd.DataFrame(columns=["serie", "segundo", "itens", "produtos", "só produtos iguais", "pagamento"])
    return (rep.groupby(["serie", "segundo"])
            .agg(itens=("produto", "size"), produtos=("produto", lambda x: " + ".join(x)),
                 **{"só produtos iguais": ("produto", lambda x: x.nunique() == 1)}, pagamento=("cash_type", "first"))
            .reset_index())


def recompras_rapidas(base: BaseDados, janela_s: float = 300) -> tuple[pd.DataFrame, dict]:
    """Máquina 1: vendas consecutivas do mesmo cartão em até ``janela_s`` segundos.

    Compara o intervalo das recompras do **mesmo produto** com o das recompras de **produto
    diferente** (que não podem ser registro em dobro) e com o intervalo mínimo entre clientes
    diferentes (o ciclo da máquina).

    Returns:
        (uma linha por recompra com intervalo e tipo, resumo com os testes).
    """
    g = base.transacoes[base.transacoes["serie"] == "Máquina 1"].sort_values("datetime").copy()
    g["intervalo (s)"] = g["datetime"].diff().dt.total_seconds()
    mesmo_cartao = (g["card"].eq(g["card"].shift()) & g["card"].notna()).fillna(False).astype(bool)
    g["tipo"] = np.where(g["produto"].eq(g["produto"].shift()).fillna(False).astype(bool), "mesmo produto", "produto diferente")
    rec = g[mesmo_cartao & (g["intervalo (s)"] <= janela_s)][["datetime", "card", "produto", "money", "intervalo (s)", "tipo"]]
    a = rec.loc[rec["tipo"] == "mesmo produto", "intervalo (s)"]
    b = rec.loc[rec["tipo"] == "produto diferente", "intervalo (s)"]
    # ciclo da máquina: menor intervalo entre duas vendas consecutivas de cartões diferentes (os dois identificados)
    cartoes_diferentes = (g["card"].notna() & g["card"].shift().notna() & ~mesmo_cartao).astype(bool)
    entre_clientes = g.loc[cartoes_diferentes, "intervalo (s)"].dropna()
    ks = stats.ks_2samp(a, b)
    resumo = {"janela (s)": janela_s, "vendas da M1": len(g),
              "recompras do mesmo produto": len(a), "% das vendas da M1": 100 * len(a) / len(g),
              "recompras de produto diferente": len(b),
              "intervalo mínimo, mesmo produto (s)": a.min(), "intervalo mediano, mesmo produto (s)": a.median(),
              "intervalo mínimo, produto diferente (s)": b.min(), "intervalo mediano, produto diferente (s)": b.median(),
              "intervalo mínimo entre clientes diferentes (s)": entre_clientes.min(),
              "recompras com intervalo < 30 s": int((rec["intervalo (s)"] < 30).sum()),
              "KS (mesmo × diferente)": ks.statistic, "p KS": ks.pvalue}
    return rec, resumo


def resumo_duplicados(base: BaseDados) -> pd.DataFrame:
    """Uma linha por máquina: duplicatas exatas, registros no mesmo segundo e participação nas vendas."""
    grupos = pedidos_mesmo_segundo(base)
    linhas = []
    for serie, t in base.transacoes.groupby("serie"):
        gs = grupos[grupos["serie"] == serie]
        extra = t.duplicated(CHAVE_EXATA)
        n_ex = int(extra.sum())
        linhas.append({"série": serie, "vendas": len(t), "duplicatas exatas (registros a mais)": n_ex,
                       "% das vendas": 100 * n_ex / len(t), "grupos no mesmo segundo": len(gs),
                       "registros nesses grupos": int(gs["itens"].sum()) if len(gs) else 0,
                       "grupos com produtos diferentes": int((~gs["só produtos iguais"].astype(bool)).sum()) if len(gs) else 0,
                       "duplicatas exatas dentro desses grupos": int((extra & t["datetime"].dt.floor("s").isin(gs["segundo"])).sum())})
    return pd.DataFrame(linhas).set_index("série")


def grafico_recompras(base: BaseDados, janela_s: float = 300) -> plt.Figure:
    """Distribuição acumulada do intervalo das recompras da M1: mesmo produto × produto diferente."""
    rec, res = recompras_rapidas(base, janela_s)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    estilos = {"mesmo produto": dict(color=cor_serie("Máquina 1"), ls="-"), "produto diferente": dict(color=CORES["contexto"], ls="--")}
    for tipo, sub in rec.groupby("tipo"):
        x = np.sort(sub["intervalo (s)"].to_numpy())
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", lw=2, label=f"{tipo} (n = {len(x)})", **estilos[tipo])
    ax.axvspan(0, 30, color=CORES["fundo_teste"], zorder=0)
    ax.annotate(f"< 30 s: onde ficaria um\nregistro em dobro\n({res['recompras com intervalo < 30 s']} recompras)", (15, 0.55), ha="center",
                fontsize=8.5, color=CORES["texto_secundario"])
    ciclo = res["intervalo mínimo entre clientes diferentes (s)"]
    ax.axvline(ciclo, color=CORES["quebra"], lw=1, ls=":")
    ax.annotate(f"menor intervalo entre\nclientes diferentes: {ciclo:.0f} s", (ciclo, 0.9), xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, color=CORES["texto_secundario"])
    ax.set_xlim(0, 240)
    ax.set_xlabel("segundos desde a compra anterior do mesmo cartão")
    ax.set_ylabel("proporção acumulada")
    ax.set_title("Máquina 1: recompras levam ~1 min, o tempo de uma nova preparação")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
