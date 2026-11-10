"""Comparação de modelos, diagnóstico de resíduos e seleção final.

**Critério de seleção final (por série × alvo).** Não é só o menor erro numa janela nem só o
teste estatístico. Combina, com pesos declarados em ``config.yaml``:

| Componente | Peso | Medida |
|---|---|---|
| Desempenho no teste | 0,40 | posto do MAE nos 14 dias de teste |
| Desempenho no backtest | 0,35 | posto do MASE médio nas 11 origens (na M2, sem histórico para backtest, usa o posto do MAE de validação interna) |
| Estabilidade | 0,25 | posto do % de origens em que o modelo bate a Média 4 semanas (na M2, posto do viés absoluto no teste) |

e um filtro de **coerência de negócio**: modelos com viés |ME| / média do teste acima do limite
(15%) são desclassificados se houver alternativa. O menor escore vence.

**Ressalva registrada:** como o teste entra na escolha (exigência do enunciado), o erro de teste
do modelo escolhido deixa de ser uma estimativa sem viés do erro futuro; o backtest de 11 origens é
a referência mais honesta de desempenho esperado.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

from src.data.preprocessing import BaseDados
from src.models.model_selection import ResultadoModelo
from src.settings import carregar_config
from src.visualization.plots import CORES, CORES_MODELOS, DIVERGENTE, SEQUENCIAL, cor_serie, formatar_eixo_datas
from src.validation.metrics import piso_ruido

_CFG = carregar_config()
ORDEM_MODELOS = list(_CFG["visualizacao"]["cores_modelos"])


# ============================================================================ significância
def diebold_mariano(erros_a, erros_b, defasagens: int = 2) -> tuple[float, float]:
    """Diebold-Mariano com correção de Harvey-Leybourne-Newbold (perda absoluta, variância Newey-West)."""
    d = np.abs(erros_a) - np.abs(erros_b)
    n = len(d)
    dc = d - d.mean()
    var = np.sum(dc ** 2) / n
    for k in range(1, defasagens + 1):
        var += 2 * (1 - k / (defasagens + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    if var <= 0:
        return np.nan, np.nan
    est = d.mean() / np.sqrt(var / n) * np.sqrt((n + 1 - 2 + 0) / n)
    return float(est), float(2 * stats.t.sf(abs(est), df=n - 1))


def tabela_dm(resultados: list[ResultadoModelo], base: BaseDados, referencia: str = "Naive sazonal") -> pd.DataFrame:
    """DM (HLN) de cada modelo contra a referência no teste, com Holm por série × alvo."""
    ref = {(r.serie, r.alvo): r for r in resultados if r.modelo == referencia}
    linhas = []
    for r in resultados:
        if r.modelo == referencia or (r.serie, r.alvo) not in ref:
            continue
        y = base.series[(r.serie, r.alvo)].loc[r.previsao.index]
        est, p = diebold_mariano((r.previsao - y).to_numpy(), (ref[(r.serie, r.alvo)].previsao - y).to_numpy())
        linhas.append({"modelo": r.modelo, "série": r.serie, "alvo": r.alvo, "DM (HLN)": est, "p": p,
                       "ΔMAE vs referência": np.mean(np.abs(r.previsao - y)) - np.mean(np.abs(ref[(r.serie, r.alvo)].previsao - y))})
    t = pd.DataFrame(linhas)
    t["p Holm"] = t.groupby(["série", "alvo"])["p"].transform(lambda p: multipletests(p.fillna(1), method="holm")[1])
    t["conclusão (5%)"] = np.where(t["p Holm"] < 0.05, np.where(t["ΔMAE vs referência"] < 0, "melhor", "pior"), "indistinguível")
    return t


def resumo_backtest(tab: pd.DataFrame) -> pd.DataFrame:
    """Por série × alvo × modelo: MASE/MAE/ME médios, posto médio, vitórias e teste t por origem (Holm).

    O diferencial de MAE por origem (blocos de 14 dias sem sobreposição) contra o naive sazonal e
    contra a Média 4 semanas passa por um teste t bilateral; os p-valores são corrigidos por Holm
    dentro de cada série × alvo.
    """
    tab = tab.copy()
    tab["posto"] = tab.groupby(["origem", "série", "alvo"])["MAE"].rank(method="min")
    saida = []
    for (serie, alvo), g in tab.groupby(["série", "alvo"]):
        refs = {nome: g[g["modelo"] == nome].set_index("origem")["MAE"] for nome in ("Naive sazonal", "Média 4 semanas")}
        linhas = []
        for modelo, gm in g.groupby("modelo"):
            gm = gm.set_index("origem")
            linha = {"série": serie, "alvo": alvo, "modelo": modelo, "origens": len(gm), "MASE médio": gm["MASE"].mean(),
                     "MAE médio": gm["MAE"].mean(), "WAPE médio %": gm["WAPE_%"].mean(), "ME médio": gm["ME"].mean(),
                     "posto médio": gm["posto"].mean(), "DP do posto": gm["posto"].std()}
            for nome, ref in refs.items():
                d = (gm["MAE"] - ref.reindex(gm.index)).dropna()
                curto = "naive" if nome == "Naive sazonal" else "média 4 sem."
                linha[f"vence {curto} (% origens)"] = 100 * (d < 0).mean() if modelo != nome else np.nan
                if modelo != nome and len(d) > 1 and d.std(ddof=1) > 0:
                    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                    linha[f"p vs {curto}"] = 2 * stats.t.sf(abs(t), df=len(d) - 1)
                else:
                    linha[f"p vs {curto}"] = np.nan
            linhas.append(linha)
        t = pd.DataFrame(linhas)
        for col in ("p vs naive", "p vs média 4 sem."):
            ok = t[col].notna()
            t.loc[ok, col.replace("p ", "p Holm ")] = multipletests(t.loc[ok, col], method="holm")[1]
        saida.append(t.sort_values("MASE médio"))
    return pd.concat(saida, ignore_index=True)


# ============================================================================ máquinas e total
def comparar_bottom_up(resultados: list[ResultadoModelo], base: BaseDados) -> pd.DataFrame:
    """Total previsto diretamente × soma das previsões de M1 e M2 do mesmo modelo (no teste)."""
    idx = {(r.modelo, r.serie, r.alvo): r for r in resultados}
    linhas = []
    for (modelo, serie, alvo), rt in idx.items():
        if serie != "Total" or (modelo, "Máquina 1", alvo) not in idx or (modelo, "Máquina 2", alvo) not in idx:
            continue
        y = base.series[("Total", alvo)].loc[rt.previsao.index]
        soma = idx[(modelo, "Máquina 1", alvo)].previsao + idx[(modelo, "Máquina 2", alvo)].previsao
        linhas.append({"modelo": modelo, "alvo": alvo, "MAE Total direto": np.mean(np.abs(rt.previsao - y)),
                       "MAE soma M1 + M2": np.mean(np.abs(soma - y)), "ME Total direto": np.mean(rt.previsao - y),
                       "ME soma M1 + M2": np.mean(soma - y)})
    t = pd.DataFrame(linhas)
    t["soma melhor?"] = t["MAE soma M1 + M2"] < t["MAE Total direto"]
    t["variação % (soma vs direto)"] = 100 * (t["MAE soma M1 + M2"] / t["MAE Total direto"] - 1)
    return t


# ============================================================================ treino × validação × teste
def _media_dobras(y: pd.Series, fim: pd.Timestamp, inicio: pd.Timestamp) -> float:
    """Média observada nos dias das dobras internas (denominador do WAPE de validação)."""
    d = _CFG["divisao"]
    origens = [fim - pd.Timedelta(days=d["horizonte"] * k) for k in range(d["n_dobras_validacao"], 0, -1)]
    origens = [o for o in origens if (o - inicio).days + 1 >= d["min_dias_treino_validacao"]]
    datas = [o + pd.Timedelta(days=h) for o in origens for h in range(1, d["horizonte"] + 1)]
    return float(y.loc[datas].mean())


def tabela_treino_validacao_teste(resultados: list[ResultadoModelo], base: BaseDados, resumo_bt: pd.DataFrame | None) -> pd.DataFrame:
    """WAPE de ajuste no treino (últimos 28 dias) com o piso de ruído, MAE de validação, MAE de teste e MASE do backtest."""
    linhas = []
    for r in resultados:
        y = base.series[(r.serie, r.alvo)]
        fim = r.origem
        ini = fim - pd.Timedelta(days=_CFG["divisao"]["dias_ajuste_recente"] - 1)
        aj = r.ajustado.loc[ini:fim]
        ok = aj.notna()
        wape_aj = (aj[ok] - y.loc[aj[ok].index]).abs().sum() / y.loc[aj[ok].index].sum() * 100 if ok.sum() > 7 else np.nan
        tickets = base.transacoes.loc[(base.transacoes["serie"] == r.serie) | (r.serie == "Total"), "money"].to_numpy(float)
        _, piso = piso_ruido(base.series[(r.serie, "demanda")].loc[:fim], tickets, ini, fim, r.alvo)
        teste = base.series[(r.serie, r.alvo)].loc[r.previsao.index]
        linha = {"série": r.serie, "alvo": r.alvo, "modelo": r.modelo,
                 "treino: WAPE ajuste % (28 d)": wape_aj, "piso de ruído %": piso,
                 "validação: MAE": r.mae_validacao, "teste: MAE": float(np.mean(np.abs(r.previsao - teste))),
                 "validação: WAPE %": 100 * r.mae_validacao / _media_dobras(y, fim, base.inicio[r.serie]) if np.isfinite(r.mae_validacao) else np.nan,
                 "teste: WAPE %": float(np.abs(r.previsao - teste).sum() / teste.sum() * 100)}
        if resumo_bt is not None:
            b = resumo_bt[(resumo_bt["série"] == r.serie) & (resumo_bt["alvo"] == r.alvo) & (resumo_bt["modelo"] == r.modelo)]
            linha["backtest: MASE médio"] = b["MASE médio"].iloc[0] if len(b) else np.nan
            linha["backtest: WAPE médio %"] = b["WAPE médio %"].iloc[0] if len(b) else np.nan
        linhas.append(linha)
    return pd.DataFrame(linhas)


# ============================================================================ resíduos
def diagnostico_residuos(r: ResultadoModelo, base: BaseDados) -> dict:
    """Resíduos de ajuste (treino) e erros de teste: média, Ljung-Box, normalidade e padrão semanal."""
    y = base.series[(r.serie, r.alvo)]
    res = (y.loc[r.ajustado.dropna().index] - r.ajustado.dropna())
    res = res.iloc[7:] if len(res) > 14 else res
    erros_teste = (y.loc[r.previsao.index] - r.previsao)
    saida = {"série": r.serie, "alvo": r.alvo, "modelo": r.modelo, "n resíduos": len(res),
             "média dos resíduos": res.mean() if len(res) else np.nan}
    if len(res) >= 20:
        lags = min(14, len(res) // 3)
        saida["Ljung-Box p (treino)"] = float(acorr_ljungbox(res, lags=[lags])["lb_pvalue"].iloc[0])
        saida["Shapiro p (treino)"] = float(stats.shapiro(res).pvalue)
        saida["Kruskal p (resíduo × dia da semana)"] = float(stats.kruskal(*[g.to_numpy() for _, g in res.groupby(res.index.dayofweek)]).pvalue)
    saida["Ljung-Box p (erros de teste)"] = float(acorr_ljungbox(erros_teste, lags=[5])["lb_pvalue"].iloc[0])
    saida["viés no teste (obs − prev)"] = float(erros_teste.mean())
    saida["t p (viés no teste = 0)"] = float(stats.ttest_1samp(erros_teste, 0).pvalue)
    return saida


def erros_por_periodo(tab_bt: pd.DataFrame) -> pd.DataFrame:
    """MASE médio por modelo antes e depois da quebra de fev/2025 (capacidade de capturar a mudança)."""
    t = tab_bt.copy()
    t["período"] = np.where(t["origem"] < pd.Timestamp("2025-01-26"), "origens antes da quebra de 2025",
                            "origens em torno/depois da quebra de 2025")
    return t.pivot_table(index=["série", "alvo", "modelo"], columns="período", values="MASE", aggfunc="mean")


# ============================================================================ seleção final
def selecao_final(tab_teste: pd.DataFrame, resumo_bt: pd.DataFrame | None) -> pd.DataFrame:
    """Escore ponderado por série × alvo (ver docstring do módulo); menor escore = escolhido."""
    pesos = _CFG["selecao"]["pesos"]
    limite = _CFG["selecao"]["limite_vies_relativo"] * 100
    saida = []
    for (serie, alvo), g in tab_teste.groupby(["série", "alvo"]):
        g = g.set_index("modelo").copy()
        g["posto teste"] = g["MAE"].rank(method="min")
        bt = None if resumo_bt is None else resumo_bt[(resumo_bt["série"] == serie) & (resumo_bt["alvo"] == alvo)].set_index("modelo")
        if bt is not None and len(bt):
            g["posto backtest"] = bt["MASE médio"].reindex(g.index).rank(method="min")
            estab = bt["vence média 4 sem. (% origens)"].reindex(g.index).fillna(50)
            g["posto estabilidade"] = (-estab).rank(method="min")
            g["base do escore"] = "teste + backtest (11 origens) + estabilidade"
        else:
            g["posto backtest"] = g["MAE validação"].rank(method="min")
            g["posto estabilidade"] = g["ME_relativo_%"].abs().rank(method="min")
            g["base do escore"] = "teste + validação interna + |viés| (sem histórico para backtest)"
        # componente indisponível (ex.: validação sem dobras) recebe o pior posto, nunca escore vazio
        for col in ("posto teste", "posto backtest", "posto estabilidade"):
            g[col] = g[col].fillna(len(g))
        g["escore"] = (pesos["posto_teste"] * g["posto teste"] + pesos["posto_backtest"] * g["posto backtest"]
                       + pesos["estabilidade"] * g["posto estabilidade"])
        g["viés relativo %"] = g["ME_relativo_%"]
        g["passa filtro de viés"] = g["viés relativo %"].abs() <= limite
        elegiveis = g[g["passa filtro de viés"]] if g["passa filtro de viés"].any() else g
        escolhido = elegiveis["escore"].idxmin()
        g["escolhido"] = g.index == escolhido
        saida.append(g.reset_index()[["série", "alvo", "modelo", "MAE", "WAPE_%", "viés relativo %", "posto teste", "posto backtest",
                                      "posto estabilidade", "escore", "passa filtro de viés", "escolhido", "base do escore"]])
    return pd.concat(saida, ignore_index=True).sort_values(["série", "alvo", "escore"])


# ============================================================================ gráficos
def grafico_heatmap_teste(tab_teste: pd.DataFrame) -> plt.Figure:
    """WAPE no teste e viés relativo (ME ÷ média do teste) por modelo e série."""
    ordem = [m for m in ORDEM_MODELOS if m in tab_teste["modelo"].unique()]
    w = tab_teste.pivot_table(index="modelo", columns=["série", "alvo"], values="WAPE_%").loc[ordem]
    v = tab_teste.pivot_table(index="modelo", columns=["série", "alvo"], values="ME_relativo_%").loc[ordem]
    fig, eixos = plt.subplots(1, 2, figsize=(17, 4.6), sharey=True)
    sns.heatmap(w, annot=True, fmt=".0f", cmap=SEQUENCIAL, ax=eixos[0], linewidths=0.6, linecolor="white",
                cbar_kws={"label": "WAPE (%)"})
    eixos[0].set_title("WAPE no teste (%) — menor é melhor")
    lim = float(np.nanmax(np.abs(v.to_numpy())))
    sns.heatmap(v, annot=True, fmt="+.0f", cmap=DIVERGENTE, center=0, vmin=-lim, vmax=lim, ax=eixos[1], linewidths=0.6,
                linecolor="white", cbar_kws={"label": "viés (ME ÷ média do teste, %)"})
    eixos[1].set_title("Viés relativo no teste — vermelho = superestima")
    for ax in eixos:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([f"{s}\n{a}" for s, a in w.columns], rotation=0, fontsize=8.5)
    fig.suptitle("Teste (10/03 a 23/03/2025): 6 modelos × 6 séries")
    fig.tight_layout()
    return fig


def grafico_backtest(resumo_bt: pd.DataFrame) -> plt.Figure:
    """MASE médio nas 11 origens por modelo (M1 e Total; baselines em cinza)."""
    combos = list(resumo_bt.groupby(["série", "alvo"]).groups)
    fig, eixos = plt.subplots(1, len(combos), figsize=(4.4 * len(combos), 4.2), sharey=True)
    ordem = [m for m in ORDEM_MODELOS if m in resumo_bt["modelo"].unique()]
    for ax, (serie, alvo) in zip(np.atleast_1d(eixos), combos):
        t = resumo_bt[(resumo_bt["série"] == serie) & (resumo_bt["alvo"] == alvo)].set_index("modelo").reindex(ordem)
        ax.barh(t.index, t["MASE médio"], color=[CORES_MODELOS[m] for m in t.index], height=0.62)
        for k, (m, r) in enumerate(t.iterrows()):
            marca = "*" if r.get("p Holm vs naive", 1) < 0.05 else ""
            ax.annotate(f"{r['MASE médio']:.2f}{marca}", (r["MASE médio"], k), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=8.5)
        ax.axvline(1, color=CORES["quebra"], ls="--", lw=1)
        ax.set_title(f"{serie} — {alvo}", fontsize=11)
        ax.set_xlabel("MASE médio (11 origens)")
        ax.set_xlim(0, t["MASE médio"].max() * 1.25)
    np.atleast_1d(eixos)[0].invert_yaxis()
    fig.suptitle("Backtest em 11 origens de 14 dias (* = melhor que o naive, teste t com Holm a 5%)")
    fig.tight_layout()
    return fig


def grafico_backtest_origens(tab_bt: pd.DataFrame, serie: str, alvo: str) -> plt.Figure:
    """MASE por origem e modelo: estabilidade ao longo do tempo e reação à quebra de 2025."""
    t = tab_bt[(tab_bt["série"] == serie) & (tab_bt["alvo"] == alvo)]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for m in [m for m in ORDEM_MODELOS if m in t["modelo"].unique()]:
        g = t[t["modelo"] == m].sort_values("origem")
        ls = "--" if m in ("Naive sazonal", "Média 4 semanas") else "-"
        ax.plot(g["origem"] + pd.Timedelta(days=7), g["MASE"], marker="o", ms=5, ls=ls, color=CORES_MODELOS[m], label=m)
    ax.axvline(pd.Timestamp("2025-01-30"), color=CORES["quebra"], ls=":", lw=1.2)
    ax.annotate("quebra de 2025", xy=(pd.Timestamp("2025-01-30"), 0.97), xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", fontsize=8.5, color=CORES["texto_secundario"], va="top")
    ax.set_ylabel("MASE no bloco de 14 dias")
    ax.set_title(f"{serie} — {alvo}: erro por bloco de previsão (centro do bloco no eixo x)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    formatar_eixo_datas(ax, curto=True)
    fig.tight_layout()
    return fig


def grafico_bottom_up(bu: pd.DataFrame) -> plt.Figure:
    """Total direto × soma das máquinas: abaixo da diagonal = soma melhor."""
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
    for ax, alvo in zip(eixos, ("demanda", "receita")):
        t = bu[bu["alvo"] == alvo]
        for _, r in t.iterrows():
            ax.scatter(r["MAE Total direto"], r["MAE soma M1 + M2"], s=90, color=CORES_MODELOS[r["modelo"]],
                       edgecolor="white", lw=1.5, zorder=3, label=r["modelo"])
            ax.annotate(r["modelo"], (r["MAE Total direto"], r["MAE soma M1 + M2"]), xytext=(5, 4),
                        textcoords="offset points", fontsize=8, color=CORES["texto_secundario"])
        lims = [min(t[["MAE Total direto", "MAE soma M1 + M2"]].min()) * 0.93, max(t[["MAE Total direto", "MAE soma M1 + M2"]].max()) * 1.07]
        ax.plot(lims, lims, color=CORES["contexto"], ls="--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("MAE — Total previsto diretamente")
        ax.set_ylabel("MAE — soma Máquina 1 + Máquina 2")
        ax.set_title(f"Total — {alvo}")
    eixos[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Total: modelo independente × soma das previsões individuais (abaixo da diagonal = soma melhor)")
    fig.tight_layout()
    return fig


def grafico_residuos(r: ResultadoModelo, base: BaseDados) -> plt.Figure:
    """Painel de resíduos do modelo escolhido: no tempo, ACF, histograma e por dia da semana."""
    from statsmodels.graphics.tsaplots import plot_acf

    y = base.series[(r.serie, r.alvo)]
    res = (y.loc[r.ajustado.dropna().index] - r.ajustado.dropna())
    erros = y.loc[r.previsao.index] - r.previsao
    fig, eixos = plt.subplots(1, 4, figsize=(19, 3.8), gridspec_kw={"width_ratios": [2, 1.1, 1, 1.1]})
    ax = eixos[0]
    ax.plot(res.index, res.values, color=cor_serie(r.serie), lw=1, label="resíduo de ajuste (treino)")
    ax.plot(erros.index, erros.values, color=CORES["observado"], marker="o", ms=4, lw=1, label="erro no teste")
    ax.axhline(0, color=CORES["eixo"], lw=1)
    ax.set_title(f"{r.modelo} — {r.serie} — {r.alvo}: resíduos (observado − previsto)")
    ax.legend(loc="upper left", fontsize=8)
    formatar_eixo_datas(ax, curto=True)
    if len(res) > 20:
        plot_acf(res, lags=min(21, len(res) // 2 - 1), ax=eixos[1], color=cor_serie(r.serie), vlines_kwargs={"colors": cor_serie(r.serie)})
    eixos[1].set_title("ACF dos resíduos de ajuste")
    sns.histplot(res, ax=eixos[2], color=cor_serie(r.serie), edgecolor="white")
    eixos[2].set_title("Distribuição dos resíduos")
    df_dow = pd.DataFrame({"res": res.values, "dia": [["seg", "ter", "qua", "qui", "sex", "sáb", "dom"][d] for d in res.index.dayofweek]})
    sns.boxplot(df_dow, x="dia", y="res", ax=eixos[3], color=cor_serie(r.serie), fliersize=2,
                order=["seg", "ter", "qua", "qui", "sex", "sáb", "dom"])
    eixos[3].axhline(0, color=CORES["eixo"], lw=1)
    eixos[3].set_title("Resíduos por dia da semana")
    eixos[3].set_xlabel("")
    fig.tight_layout()
    return fig


def grafico_treino_validacao_teste(tvt: pd.DataFrame) -> plt.Figure:
    """WAPE no treino (ajuste), validação e teste por modelo, com o piso de ruído (M1 e Total, demanda)."""
    combos = [("Máquina 1", "demanda"), ("Máquina 2", "demanda"), ("Total", "demanda")]
    fig, eixos = plt.subplots(1, 3, figsize=(18, 4.6), sharey=False)
    ordem = [m for m in ORDEM_MODELOS if m in tvt["modelo"].unique()]
    for ax, (serie, alvo) in zip(eixos, combos):
        t = tvt[(tvt["série"] == serie) & (tvt["alvo"] == alvo)].set_index("modelo").reindex(ordem)
        x = np.arange(len(t))
        for k, (col, cor, rot) in enumerate([("treino: WAPE ajuste % (28 d)", CORES["contexto"], "treino (ajuste, 28 dias)"),
                                             ("validação: WAPE %", CORES["Máquina 1"], "validação interna"),
                                             ("teste: WAPE %", CORES["observado"], "teste")]):
            ax.bar(x + (k - 1) * 0.27, t[col], width=0.25, color=cor, label=rot)
        ax.axhline(t["piso de ruído %"].iloc[0], color=CORES["Máquina 2"], ls="--", lw=1.3, label="piso de ruído (Poisson)")
        ax.set_xticks(x)
        ax.set_xticklabels(t.index, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("WAPE (%)")
        ax.set_title(f"{serie} — {alvo}")
    eixos[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Treino × validação × teste (ajuste abaixo do piso = sobreajuste, não mérito)")
    fig.tight_layout()
    return fig
