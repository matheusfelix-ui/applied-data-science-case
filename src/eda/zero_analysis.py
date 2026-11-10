"""O que significam os dias com zero vendas?

Um zero dentro do período ativo pode ser (a) um dia de demanda baixa em que, por acaso, ninguém
comprou, ou (b) uma situação operacional (máquina fechada, sem energia, falha de registro).
Para separar as duas leituras, comparamos cada zero com o que a demanda local tornaria plausível:

    P(Y = 0 | λ) = exp(−λ),  com λ = média do mesmo dia da semana nas 4 semanas anteriores e
    nas 4 posteriores (sem o próprio dia), o que já incorpora o padrão semanal.

Se P(Y = 0) é razoável (≥ 1%), o zero é compatível com demanda baixa. Se é minúscula, um zero
"natural" é implausível e a hipótese operacional (fechamento/falha) ganha força. Feriados e
sequências de zeros também são verificados. A decisão de tratamento é a mesma nos dois casos:
**manter zero**, porque zero transações implica receita zero e porque não há como saber a demanda
que existiria se a máquina estivesse operando (imputar inventaria vendas).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.preprocessing import BaseDados
from src.settings import carregar_config
from src.visualization.plots import CORES, DIAS_PT, cor_serie, formatar_eixo_datas


def nivel_local_sem_dia(y: pd.Series, semanas: int = 4) -> pd.Series:
    """Média dos mesmos dias da semana nas ``semanas`` vizinhas (antes e depois), sem o próprio dia."""
    vizinhos = pd.concat([y.shift(7 * k) for k in range(-semanas, semanas + 1) if k != 0], axis=1)
    return vizinhos.mean(axis=1)


def tabela_zeros(base: BaseDados) -> pd.DataFrame:
    """Lista dos dias sem venda, com contexto e probabilidade de zero sob Poisson(λ local)."""
    cal = carregar_config()["calendario"]
    datas_especiais = {**cal["feriados"], **cal["datas_comerciais"]}
    linhas = []
    for serie in ("Máquina 1", "Máquina 2"):
        y = base.series[(serie, "demanda")]
        lam = nivel_local_sem_dia(y)
        zeros = y.index[y == 0]
        sequencia = (y == 0).astype(int)
        grupos = (sequencia.diff() != 0).cumsum()
        tamanho = sequencia.groupby(grupos).transform("sum")
        for d in zeros:
            p0 = float(np.exp(-lam.loc[d])) if pd.notna(lam.loc[d]) else np.nan
            linhas.append({
                "série": serie, "data": d, "dia da semana": DIAS_PT[d.dayofweek],
                "data especial": datas_especiais.get(f"{d:%m-%d}", "—"),
                "dias zerados consecutivos": int(tamanho.loc[d]),
                "λ local (mesmo dia da semana, ±4 sem.)": lam.loc[d],
                "P(Y = 0 | λ)": p0,
                "leitura": ("compatível com demanda baixa" if p0 >= 0.01 else
                            "implausível como acaso → provável fechamento/falha operacional") if pd.notna(p0) else "sem vizinhança",
            })
    return pd.DataFrame(linhas)


def resumo_zeros(base: BaseDados) -> pd.DataFrame:
    """Quantidade de zeros e maior sequência por série."""
    linhas = []
    for serie in ("Máquina 1", "Máquina 2", "Total"):
        y = base.series[(serie, "demanda")]
        z = (y == 0).astype(int)
        maior = int(z.groupby((z.diff() != 0).cumsum()).sum().max())
        linhas.append({"série": serie, "dias": len(y), "dias com zero vendas": int(z.sum()),
                       "% de dias zerados": 100 * z.mean(), "maior sequência de zeros": maior,
                       "receita zero ⇔ demanda zero": bool(((y == 0) == (base.series[(serie, "receita")] == 0)).all())})
    return pd.DataFrame(linhas).set_index("série")


def grafico_zeros(base: BaseDados) -> plt.Figure:
    """Linha do tempo da demanda com os dias zerados marcados e a P(Y=0) de cada um."""
    tabela = tabela_zeros(base)
    fig, eixos = plt.subplots(2, 1, figsize=(15, 7.5), gridspec_kw={"height_ratios": [1.6, 1]})
    ax = eixos[0]
    for serie in ("Máquina 1", "Máquina 2"):
        y = base.series[(serie, "demanda")]
        ax.plot(y.index, y.values, color=cor_serie(serie), lw=1, alpha=0.75, label=serie)
        z = tabela[tabela["série"] == serie]
        ax.scatter(z["data"], np.zeros(len(z)), s=70, marker="v", color=cor_serie(serie), edgecolor=CORES["observado"],
                   lw=0.8, zorder=5, label=f"{serie}: dia sem venda")
    ax.set_ylabel("transações/dia")
    ax.set_title("Dias sem venda dentro do período ativo de cada máquina")
    formatar_eixo_datas(ax)
    ax.legend(loc="upper left", ncol=2)
    ax = eixos[1]
    rotulos = [f"{r['série'][-1]} · {r['data']:%d/%m/%y} ({r['dia da semana']})" for _, r in tabela.iterrows()]
    cores = [cor_serie(s) for s in tabela["série"]]
    ax.barh(rotulos, tabela["P(Y = 0 | λ)"], color=cores, height=0.6)
    ax.axvline(0.01, color=CORES["quebra"], ls="--", lw=1.1)
    ax.annotate("limiar 1%", xy=(0.01, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -10),
                textcoords="offset points", fontsize=8.5, color=CORES["texto_secundario"])
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("P(Y = 0) sob Poisson com o nível local (escala log)")
    ax.set_title("Quão plausível seria cada zero como simples falta de clientes?")
    fig.tight_layout()
    return fig
