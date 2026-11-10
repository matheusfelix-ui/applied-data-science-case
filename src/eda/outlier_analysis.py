"""Outliers: identificação por vários critérios, investigação de causas e impacto na modelagem.

Critérios aplicados às séries diárias (demanda e receita):

| Critério | Regra | Por que |
|---|---|---|
| IQR | fora de [Q1 − 1,5·IQR, Q3 + 1,5·IQR] | clássico, mas ignora tendência e padrão semanal |
| z robusto (MAD) | \\|y − mediana\\| / (1,4826·MAD) > 3,5 | resistente aos próprios outliers |
| z-score | \\|z\\| > 3 | referência (supõe normalidade, que as contagens não têm) |
| Condicional ao dia da semana | z robusto > 3,5 dentro de cada dia da semana | sábado baixo não é anômalo |
| Resíduo STL | resíduo da decomposição STL robusta (período 7) com z robusto > 3,5 | desconta nível local e sazonalidade |

Um dia é considerado **atípico** quando ao menos 2 critérios que respeitam a estrutura da série
(condicional e STL) ou 3 critérios no total o marcam. A causa é investigada (proximidade de quebra,
calendário, consistência das transações) e **nada é removido**: um pico legítimo é informação
sobre a variabilidade real que o modelo precisa representar nos intervalos.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from src.data.preprocessing import BaseDados
from src.settings import carregar_config
from src.visualization.plots import CORES, cor_serie, formatar_eixo_datas, anotar_quebras

LIMIAR_Z_ROBUSTO = 3.5


def z_robusto(x: pd.Series) -> pd.Series:
    """(x − mediana) / (1,4826 · MAD); MAD zero vira NaN para evitar divisão por zero."""
    mad = (x - x.median()).abs().median()
    return (x - x.median()) / (1.4826 * mad) if mad > 0 else pd.Series(np.nan, index=x.index)


def marcar_outliers(y: pd.Series, quebras: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    """Aplica os cinco critérios a uma série diária e devolve as marcações por dia.

    Com ``quebras``, o critério condicional é calculado dentro de cada combinação regime × dia da
    semana: sem isso, um novo patamar (ex.: fev/2025) faria todos os dias altos do regime novo
    parecerem atípicos frente às medianas do regime antigo.
    """
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    marc = pd.DataFrame(index=y.index)
    marc["valor"] = y
    marc["IQR"] = (y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)
    marc["MAD"] = z_robusto(y).abs() > LIMIAR_Z_ROBUSTO
    marc["z-score"] = ((y - y.mean()) / y.std()).abs() > 3
    regime = sum((y.index >= q).astype(int) for q in (quebras or [])) if quebras else np.zeros(len(y), dtype=int)
    grupos = [np.asarray(regime), y.index.dayofweek]
    marc["condicional (dia da semana)"] = y.groupby(grupos).transform(z_robusto).abs() > LIMIAR_Z_ROBUSTO
    if len(y) >= 28:
        stl = STL(y, period=7, robust=True).fit()
        marc["resíduo STL"] = z_robusto(stl.resid).abs() > LIMIAR_Z_ROBUSTO
        marc["tendência STL"] = stl.trend
    else:
        marc["resíduo STL"] = False
        marc["tendência STL"] = np.nan
    criterios = ["IQR", "MAD", "z-score", "condicional (dia da semana)", "resíduo STL"]
    marc["nº de critérios"] = marc[criterios].sum(axis=1)
    estruturais = marc[["condicional (dia da semana)", "resíduo STL"]].sum(axis=1)
    marc["atípico"] = (estruturais >= 2) | (marc["nº de critérios"] >= 3)
    return marc


def investigar(base: BaseDados, serie: str, alvo: str, quebras: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    """Classifica cada dia atípico: quebra de nível, calendário, evento legítimo ou inconsistência.

    Inconsistência = transações do dia com preço fora da tabela do produto naquele mês ou
    timestamps duplicados com mesmo produto e valor (indício de registro em dobro).
    """
    cal = carregar_config()["calendario"]
    especiais = {**cal["feriados"], **cal["datas_comerciais"]}
    marc = marcar_outliers(base.series[(serie, alvo)], quebras)
    atip = marc[marc["atípico"]].copy()
    t = base.transacoes[base.transacoes["serie"] == serie] if serie != "Total" else base.transacoes
    precos_mes = t.groupby([t["date"].dt.to_period("M"), "produto"])["money"].agg(lambda s: set(s.round(2)))
    linhas = []
    for d, r in atip.iterrows():
        dia = t[t["date"] == d]
        duplicados = int(dia.duplicated(subset=["datetime", "produto", "money"]).sum())
        perto_quebra = [q for q in (quebras or []) if abs((d - q).days) <= 14]
        if perto_quebra:
            causa = f"até 14 dias de uma quebra de nível ({perto_quebra[0]:%d/%m/%Y})"
        elif f"{d:%m-%d}" in especiais:
            causa = f"data especial: {especiais[f'{d:%m-%d}']}"
        elif duplicados > 0 and serie == "Máquina 1":
            causa = "possível inconsistência: registros repetidos no mesmo instante"
        else:
            causa = "evento legítimo (pico/vale isolado, transações consistentes)"
        tendencia = r["tendência STL"]
        linhas.append({"série": serie, "alvo": alvo, "data": d, "valor": r["valor"],
                       "nível local (tendência STL)": tendencia,
                       "desvio vs nível (%)": 100 * (r["valor"] / tendencia - 1) if pd.notna(tendencia) and tendencia > 0 else np.nan,
                       "nº de critérios": int(r["nº de critérios"]), "transações no dia": len(dia),
                       "registros repetidos": duplicados, "causa provável": causa})
    return pd.DataFrame(linhas)


def resumo_criterios(base: BaseDados, quebras: dict[str, list[pd.Timestamp]] | None = None) -> pd.DataFrame:
    """Quantos dias cada critério marca, por série e alvo (mostra o desacordo entre métodos).

    Args:
        quebras: ``{série: [datas]}`` usadas no critério condicional por regime.
    """
    linhas = []
    for (serie, alvo), y in base.series.items():
        m = marcar_outliers(y, (quebras or {}).get(serie))
        linhas.append({"série": serie, "alvo": alvo, **{c: int(m[c].sum()) for c in
                       ["IQR", "MAD", "z-score", "condicional (dia da semana)", "resíduo STL", "atípico"]}})
    return pd.DataFrame(linhas)


def sensibilidade_modelagem(base: BaseDados, serie: str = "Máquina 1", alvo: str = "demanda",
                            n_origens: int = 11, horizonte: int = 14) -> pd.DataFrame:
    """Impacto de tratar os atípicos: Média 4 semanas prevista com histórico bruto × winsorizado.

    O alvo de avaliação é sempre o dado bruto (o que realmente aconteceu). Se a winsorização
    melhorasse o erro de forma consistente, os outliers estariam distorcendo o nível; se não,
    eles são parte da variabilidade a ser representada.
    """
    y = base.series[(serie, alvo)]
    fim = y.index.max() - pd.Timedelta(days=horizonte)
    origens = [fim - pd.Timedelta(days=horizonte * k) for k in range(n_origens - 1, -1, -1)]
    linhas = []
    for o in origens:
        hist = y.loc[:o]
        marc = marcar_outliers(hist)
        tratado = hist.copy()
        lim = hist.quantile([0.01, 0.99])
        tratado[marc["atípico"]] = tratado[marc["atípico"]].clip(lim.iloc[0], lim.iloc[1])
        datas = pd.date_range(o + pd.Timedelta(days=1), periods=horizonte)
        real = y.loc[datas]
        for rotulo, h in [("bruto", hist), ("winsorizado (atípicos → p1/p99)", tratado)]:
            prev = [np.mean([h.loc[d - pd.Timedelta(days=7 * j)] for j in range(1, 5)
                             if d - pd.Timedelta(days=7 * j) <= o]) for d in datas]
            linhas.append({"origem": o, "histórico": rotulo, "MAE": np.mean(np.abs(np.array(prev) - real.to_numpy()))})
    tabela = pd.DataFrame(linhas).pivot(index="origem", columns="histórico", values="MAE")
    tabela["diferença (winsorizado − bruto)"] = tabela.iloc[:, 1] - tabela.iloc[:, 0]
    return tabela


def grafico_outliers(base: BaseDados, quebras: dict[str, dict[str, pd.Timestamp]] | None = None) -> plt.Figure:
    """Séries com nível STL e dias atípicos destacados (M1 e Total, demanda e receita)."""
    fig, eixos = plt.subplots(2, 2, figsize=(16, 7.5), sharex=True)
    for i, alvo in enumerate(("demanda", "receita")):
        for j, serie in enumerate(("Máquina 1", "Total")):
            ax = eixos[i, j]
            q = (quebras or {}).get(serie, {})
            m = marcar_outliers(base.series[(serie, alvo)], list(q.values()))
            ax.plot(m.index, m["valor"], color=CORES["contexto"], lw=0.8, label="Diário")
            ax.plot(m.index, m["tendência STL"], color=cor_serie(serie), lw=2, label="Nível (tendência STL)")
            at = m[m["atípico"]]
            ax.scatter(at.index, at["valor"], s=55, color=CORES["observado"], zorder=5, label=f"Atípico ({len(at)} dias)")
            for d, r in at.iterrows():
                ax.annotate(f"{d:%d/%m}", (d, r["valor"]), xytext=(3, 4), textcoords="offset points", fontsize=7.5,
                            color=CORES["texto_secundario"])
            if q:
                anotar_quebras(ax, q)
            ax.set_title(f"{serie} — {alvo}")
            ax.set_ylabel("transações/dia" if alvo == "demanda" else "receita/dia (u.m.)")
            formatar_eixo_datas(ax)
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Dias atípicos: consenso entre critérios que respeitam o nível local e o padrão semanal")
    fig.tight_layout()
    return fig


# ============================================================================ calendário das anomalias
def classificar_dia(d: pd.Timestamp, feriados: dict[str, str]) -> str:
    """Rótulo de calendário: feriado, dia seguinte a feriado, fim de semana ou dia útil (nessa ordem)."""
    if f"{d:%m-%d}" in feriados:
        return "feriado"
    if f"{d - pd.Timedelta(days=1):%m-%d}" in feriados:
        return "dia seguinte a feriado"
    return "fim de semana" if d.dayofweek >= 5 else "dia útil"


def calendario_anomalias(base: BaseDados, quebras: dict[str, list[pd.Timestamp]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Em que tipo de dia caem as anomalias? Quedas e picos são separados.

    Anomalias = dias atípicos (``marcar_outliers``) de qualquer série/alvo + dias sem venda da M1.
    Uma data conta uma vez. A proporção em feriado/fim de semana é comparada com a do calendário
    (teste binomial unilateral). Sem esse controle, "cai no fim de semana" não diria nada: 2 de
    cada 7 dias já são fim de semana.

    Returns:
        (uma linha por data, resumo por sentido com o teste).
    """
    from scipy import stats

    feriados = carregar_config()["calendario"]["feriados"]
    registros: dict[pd.Timestamp, dict] = {}
    for (serie, alvo), y in base.series.items():
        m = marcar_outliers(y, (quebras or {}).get(serie))
        for d, r in m[m["atípico"]].iterrows():
            sentido = "queda" if pd.notna(r["tendência STL"]) and r["valor"] < r["tendência STL"] else "pico"
            reg = registros.setdefault(d, {"data": d, "sentido": sentido, "séries": set()})
            reg["séries"].add(f"{serie} ({alvo})")
    zeros = base.diarios["Máquina 1"].index[base.diarios["Máquina 1"]["demanda"] == 0]
    for d in zeros:
        reg = registros.setdefault(d, {"data": d, "sentido": "queda", "séries": set()})
        reg["sentido"] = "queda"
        reg["séries"].add("Máquina 1 (zero vendas)")
    tabela = pd.DataFrame(registros.values()).sort_values("data").reset_index(drop=True)
    tabela["séries"] = tabela["séries"].map(lambda s: ", ".join(sorted(s)))
    tabela["dia da semana"] = tabela["data"].dt.dayofweek.map(dict(enumerate(["seg", "ter", "qua", "qui", "sex", "sáb", "dom"])))
    tabela["calendário"] = tabela["data"].map(lambda d: classificar_dia(d, feriados))
    tabela["feriado ou fim de semana"] = tabela["calendário"] != "dia útil"

    dias = base.series[("Total", "demanda")].index
    p_cal = float(np.mean([classificar_dia(d, feriados) != "dia útil" for d in dias]))
    linhas = []
    for sentido, g in tabela.groupby("sentido"):
        k, n = int(g["feriado ou fim de semana"].sum()), len(g)
        linhas.append({"sentido": sentido, "dias": n, "em feriado ou fim de semana": k, "%": 100 * k / n,
                       "% esperado pelo calendário": 100 * p_cal,
                       "p (binomial, unilateral)": stats.binomtest(k, n, p_cal, alternative="greater").pvalue})
    return tabela, pd.DataFrame(linhas)
