"""Detecção e testes formais de quebras estruturais.

**Detecção sem vazamento (reaproveitada de ``modelagens_testes.ipynb``, seção 4.2).**
- Quebras de nível: Segmentação Binária (custo L2, segmentos ≥ 28 dias) na demanda diária, com o
  número de quebras escolhido por BIC (e verificação com tamanho efetivo de amostra n·(1−r₁)/(1+r₁)).
- Mudança da janela de vendas: 1 quebra por Segmentação Binária na proporção diária de vendas fora
  de 10h–19h da Máquina 1, aceita só se o BIC a preferir a nenhuma.
Ambas são recalculadas **apenas com dados até a origem** de cada previsão (``DetectorQuebras``).

**Testes formais (novos neste projeto).** Uma data escolhida olhando os dados invalida o p-valor
de um teste de Chow comum (o máximo de muitas estatísticas não tem distribuição F). Por isso:

| Teste | H0 | Observação |
|---|---|---|
| sup-Wald de Andrews (1993) com erros HAC | nenhuma quebra de nível em [15%, 85%] da amostra | p-valor da distribuição assintótica (ponte browniana, simulada) |
| OLS-CUSUM (Ploberger & Krämer, 1992) | estabilidade dos parâmetros | resíduos de y ~ dia da semana |
| Razão de verossimilhança Poisson/NB2 | dummies de regime = 0 | modelo de contagem com dia da semana |
| Regressão segmentada (HAC) | sem salto de nível / sem mudança de inclinação | separa mudança de nível de mudança de tendência |
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.stattools import acf

from src.logging_config import obter_logger
from src.settings import carregar_config
from src.visualization.plots import CORES, anotar_quebras, cor_serie, formatar_eixo_datas, reservar_topo

log = obter_logger(__name__)
_CFG = carregar_config()["quebras"]


# ============================================================================ detecção (porte fiel)
def segmentacoes_binseg(y: pd.Series, min_segmento: int, max_quebras: int) -> list:
    """Segmentações candidatas (0 a ``max_quebras`` quebras) por Segmentação Binária (L2)."""
    algoritmo = rpt.Binseg(model="l2", min_size=min_segmento).fit(y.to_numpy().reshape(-1, 1))
    candidatas = []
    for k in range(max_quebras + 1):
        try:
            pontos = [] if k == 0 else algoritmo.predict(n_bkps=k)[:-1]
        except Exception:
            break
        candidatas.append((k, pontos, np.split(y.to_numpy(), pontos)))
    return candidatas


def detectar_quebras_nivel(y: pd.Series, n_efetivo: bool = False, cfg: dict | None = None) -> tuple[list, pd.DataFrame]:
    """Quebras de nível com o número de quebras escolhido por BIC.

    Args:
        y: série diária (só o período disponível na origem).
        n_efetivo: usa o tamanho efetivo de amostra no BIC (verificação de excesso de segmentação).

    Returns:
        (datas das quebras, tabela com BIC e médias de cada segmentação candidata).
    """
    cfg = cfg or _CFG
    y = y.dropna()
    n = len(y)
    if n < cfg["min_dias_deteccao"]:
        return [], pd.DataFrame()
    candidatas = segmentacoes_binseg(y, cfg["min_segmento"], cfg["max_quebras"])
    residuos = np.concatenate([s - s.mean() for s in candidatas[-1][2]])
    r1 = acf(residuos, nlags=1, fft=False)[1]
    n_bic = n * (1 - r1) / (1 + r1) if n_efetivo else n
    linhas, melhor = [], (np.inf, [])
    for k, pontos, segmentos in candidatas:
        rss = sum(((s - s.mean()) ** 2).sum() for s in segmentos)
        bic = n_bic * np.log(rss / n) + (2 * k + 1) * np.log(n_bic)
        datas = [y.index[i] for i in pontos]
        linhas.append({"nº de quebras": k, "datas": ", ".join(f"{d:%d/%m/%y}" for d in datas) or "—",
                       "médias dos segmentos": " → ".join(f"{s.mean():.1f}" for s in segmentos), "BIC": bic})
        if bic < melhor[0]:
            melhor = (bic, datas)
    tabela = pd.DataFrame(linhas)
    tabela.attrs["r1"] = r1
    return melhor[1], tabela


def proporcao_fora_horario(transacoes: pd.DataFrame, fim: pd.Timestamp | None = None) -> pd.Series:
    """Proporção diária das vendas da Máquina 1 fora da janela original (10h–19h)."""
    h0, h1 = _CFG["horas_nucleo_original"]
    d = transacoes[(transacoes["maquina"] == 1)]
    if fim is not None:
        d = d[d["date"] <= fim]
    return (~d["hora"].between(h0, h1)).groupby(d["date"]).mean()


def detectar_mudanca_janela(transacoes: pd.DataFrame, fim: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """Mudança da janela de vendas detectada só com dados até ``fim`` (BIC: 1 quebra × nenhuma)."""
    fora = proporcao_fora_horario(transacoes, fim)
    if len(fora) < _CFG["min_dias_deteccao"]:
        return None
    x, n = fora.to_numpy(), len(fora)
    ponto = rpt.Binseg(model="l2", min_size=_CFG["min_segmento_janela"]).fit(x.reshape(-1, 1)).predict(n_bkps=1)[0]
    rss0 = ((x - x.mean()) ** 2).sum()
    rss1 = sum(((s - s.mean()) ** 2).sum() for s in np.split(x, [ponto]))
    if n * np.log(rss1 / n) + 3 * np.log(n) < n * np.log(rss0 / n) + np.log(n):
        return fora.index[ponto]
    return None


def refinar_mudanca_janela(transacoes: pd.DataFrame, data: pd.Timestamp, raio: int = 4) -> pd.Timestamp:
    """Primeiro dia, na vizinhança da data detectada, com vendas fora de 10h–19h (análise retrospectiva).

    Reproduz o refinamento da seção 19.1 do notebook principal (data adotada: 14/05/2024). Usado
    só nas análises descritivas/causais; a modelagem usa a data detectada até a origem.
    """
    fora = proporcao_fora_horario(transacoes)
    vizinhanca = fora.loc[data - pd.Timedelta(days=raio): data + pd.Timedelta(days=raio)]
    positivos = vizinhanca.index[(vizinhanca > 0).to_numpy()]
    return positivos[0] if len(positivos) else data


@dataclass
class DetectorQuebras:
    """Quebras conhecidas em cada origem (detecção refeita só com o passado, com cache).

    Args:
        series: ``BaseDados.series``.
        transacoes: ``BaseDados.transacoes`` (para a janela de vendas).
    """
    series: dict
    transacoes: pd.DataFrame
    _cache: dict = field(default_factory=dict)

    def quebras_ate(self, serie: str, fim: pd.Timestamp) -> list[pd.Timestamp]:
        """Quebras de nível da série detectadas com dados até ``fim``."""
        chave = ("nivel", serie, fim)
        if chave not in self._cache:
            self._cache[chave] = detectar_quebras_nivel(self.series[(serie, "demanda")].loc[:fim])[0]
        return self._cache[chave]

    def janela_ate(self, serie: str, fim: pd.Timestamp) -> pd.Timestamp | None:
        """Data da extensão da janela de vendas conhecida em ``fim`` (não se aplica à Máquina 2)."""
        if serie == "Máquina 2":
            return None
        chave = ("janela", fim)
        if chave not in self._cache:
            self._cache[chave] = detectar_mudanca_janela(self.transacoes, fim)
        return self._cache[chave]

    def quebra_novos_clientes_ate(self, serie: str, fim: pd.Timestamp) -> pd.Timestamp | None:
        """Quebra final (aumento de patamar de 2025) conhecida em ``fim``.

        É a última quebra de nível detectada **com aumento de média** posterior a 01/12/2024; antes
        de ela ser detectável, devolve None (a dummy fica 0, sem vazamento).
        """
        quebras = self.quebras_ate(serie, fim)
        y = self.series[(serie, "demanda")].loc[:fim]
        candidatas = []
        for k, q in enumerate(quebras):
            anterior = quebras[k - 1] if k > 0 else y.index.min()
            seguinte = quebras[k + 1] if k + 1 < len(quebras) else fim + pd.Timedelta(days=1)
            if q >= pd.Timestamp("2024-12-01") and y.loc[q:seguinte - pd.Timedelta(days=1)].mean() > y.loc[anterior:q - pd.Timedelta(days=1)].mean():
                candidatas.append(q)
        return candidatas[-1] if candidatas else None


# ============================================================================ testes formais
def _matriz_dia_semana(indice: pd.DatetimeIndex) -> pd.DataFrame:
    X = pd.get_dummies(indice.dayofweek, prefix="dow", drop_first=True, dtype=float)
    X.index = indice
    return sm.add_constant(X)


def _simular_sup_wald(trimming: float, n_sim: int = 5000, grade: int = 1000, semente: int = 2024) -> np.ndarray:
    """Distribuição assintótica do sup-Wald (1 parâmetro) via ponte browniana: sup (B(π)−πB(1))²/(π(1−π))."""
    rng = np.random.default_rng(semente)
    passos = rng.standard_normal((n_sim, grade)) / np.sqrt(grade)
    w = np.cumsum(passos, axis=1)
    pi = np.arange(1, grade + 1) / grade
    janela = (pi >= trimming) & (pi <= 1 - trimming)
    ponte = (w - pi * w[:, [-1]])[:, janela]
    return (ponte ** 2 / (pi[janela] * (1 - pi[janela]))).max(axis=1)


_NULA_SUP_WALD: dict = {}


def teste_sup_wald(y: pd.Series, trimming: float | None = None, hac_lags: int = 7) -> dict:
    """Teste de Andrews (1993) para uma quebra de nível em data desconhecida.

    Para cada data candidata no intervalo aparado, estima y = dia da semana + δ·1{t ≥ τ} por MQO
    e calcula a estatística de Wald de δ = 0 com variância HAC (Newey-West). A estatística é o
    máximo; o p-valor vem da distribuição assintótica simulada (não da F).
    """
    trimming = trimming or _CFG["sup_f_trimming"]
    y = y.dropna()
    X0 = _matriz_dia_semana(y.index)
    n = len(y)
    inicio, fim = int(np.floor(trimming * n)), int(np.ceil((1 - trimming) * n))
    estatisticas = []
    for i in range(inicio, fim):
        X = X0.copy()
        X["quebra"] = (np.arange(n) >= i).astype(float)
        res = sm.OLS(y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        estatisticas.append((res.tvalues["quebra"] ** 2, y.index[i], res.params["quebra"]))
    wald, data, efeito = max(estatisticas, key=lambda t: t[0])
    if trimming not in _NULA_SUP_WALD:
        _NULA_SUP_WALD[trimming] = _simular_sup_wald(trimming)
    nula = _NULA_SUP_WALD[trimming]
    return {"sup-Wald": wald, "data estimada": data, "salto de nível estimado": efeito,
            "p (assintótico de Andrews)": float((nula >= wald).mean()),
            "valor crítico 5%": float(np.quantile(nula, 0.95)), "n": n,
            "curva": pd.Series([e[0] for e in estatisticas], index=[e[1] for e in estatisticas])}


def teste_cusum(y: pd.Series) -> dict:
    """OLS-CUSUM nos resíduos de y ~ dia da semana (H0: parâmetros estáveis)."""
    y = y.dropna()
    res = sm.OLS(y.to_numpy(), _matriz_dia_semana(y.index)).fit()
    estat, p, _ = breaks_cusumolsresid(res.resid, ddof=res.df_model + 1)
    return {"estatística OLS-CUSUM": float(estat), "p": float(p)}


def teste_razao_verossimilhanca(y: pd.Series, quebras: list[pd.Timestamp], familia: str = "NB2") -> dict:
    """LR das dummies de regime num modelo de contagem com dia da semana (H0: todas as dummies = 0)."""
    y = y.dropna()
    X0 = _matriz_dia_semana(y.index)
    X1 = X0.copy()
    for k, q in enumerate(quebras, start=1):
        X1[f"regime_{k}"] = (y.index >= q).astype(float)
    if familia == "Poisson":
        m0 = sm.GLM(y, X0, family=sm.families.Poisson()).fit()
        m1 = sm.GLM(y, X1, family=sm.families.Poisson()).fit()
    else:
        m0 = sm.NegativeBinomial(y, X0, loglike_method="nb2").fit(disp=0, maxiter=500)
        m1 = sm.NegativeBinomial(y, X1, loglike_method="nb2").fit(disp=0, maxiter=500)
    lr = 2 * (m1.llf - m0.llf)
    gl = len(quebras)
    irr = {c: float(np.exp(m1.params[c])) for c in X1.columns if c.startswith("regime_")}
    pvals = {c: float(m1.pvalues[c]) for c in irr}
    return {"família": familia, "LR": float(lr), "gl": gl, "p": float(stats.chi2.sf(lr, gl)), "IRR por regime": irr,
            "p por regime": pvals}


def regressao_segmentada(y: pd.Series, data: pd.Timestamp, semanas: int = 8, hac_lags: int = 7) -> dict:
    """Mudança de nível e de inclinação em torno de ``data`` (janela de ±``semanas``), com HAC.

    y_t = β0 + β1·t + β2·1{t ≥ τ} + β3·(t − τ)·1{t ≥ τ} + dia da semana + ε.
    """
    janela = y.loc[data - pd.Timedelta(weeks=semanas): data + pd.Timedelta(weeks=semanas) - pd.Timedelta(days=1)].dropna()
    X = _matriz_dia_semana(janela.index)
    t = np.arange(len(janela), dtype=float)
    pos = (janela.index >= data).astype(float)
    X["tempo"] = t
    X["nivel"] = pos
    X["inclinacao"] = (t - t[pos.argmax()]) * pos
    res = sm.OLS(janela.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return {"data": data, "dias": len(janela), "salto de nível": res.params["nivel"], "p nível": res.pvalues["nivel"],
            "mudança de inclinação (/dia)": res.params["inclinacao"], "p inclinação": res.pvalues["inclinacao"],
            "média antes": janela[pos == 0].mean(), "média depois": janela[pos == 1].mean(),
            "Mann-Whitney p": stats.mannwhitneyu(janela[pos == 0], janela[pos == 1]).pvalue}


def confirmar_quebras_localmente(y: pd.Series, quebras: list[pd.Timestamp]) -> pd.DataFrame:
    """Confirma cada quebra com um sup-Wald dentro do seu segmento local (entre as quebras vizinhas).

    É a lógica sequencial de Bai-Perron: se a quebra é real, o teste no trecho que a contém
    rejeita H0 e estima uma data próxima.
    """
    limites = [y.index.min()] + list(quebras) + [y.index.max() + pd.Timedelta(days=1)]
    linhas = []
    for k, q in enumerate(quebras, start=1):
        trecho = y.loc[limites[k - 1]: limites[k + 1] - pd.Timedelta(days=1)]
        r = teste_sup_wald(trecho)
        linhas.append({"quebra (Binseg + BIC)": q, "trecho testado": f"{trecho.index.min():%d/%m/%y}–{trecho.index.max():%d/%m/%y}",
                       "sup-Wald": r["sup-Wald"], "valor crítico 5%": r["valor crítico 5%"],
                       "p (Andrews)": r["p (assintótico de Andrews)"], "data estimada pelo sup-Wald": r["data estimada"],
                       "salto de nível (transações/dia)": r["salto de nível estimado"],
                       "média antes": y.loc[limites[k - 1]: q - pd.Timedelta(days=1)].mean(),
                       "média depois": y.loc[q: limites[k + 1] - pd.Timedelta(days=1)].mean()})
    return pd.DataFrame(linhas)


def bateria_testes(series: dict, transacoes: pd.DataFrame, fim: pd.Timestamp) -> dict:
    """Executa a detecção e os testes formais nas séries de demanda (dados até ``fim``).

    Returns:
        dict com tabelas: ``deteccao`` (BIC), ``globais`` (sup-Wald, CUSUM, LR), ``locais``
        (confirmação de cada quebra), ``segmentadas`` (nível × inclinação) e ``janela`` (mudança de horário).
    """
    detector = DetectorQuebras(series, transacoes)
    saida = {"deteccao": {}, "globais": [], "locais": {}, "segmentadas": [], "detector": detector}
    for serie in ("Máquina 1", "Total"):
        y = series[(serie, "demanda")].loc[:fim]
        datas, tabela = detectar_quebras_nivel(y)
        datas_ef, _ = detectar_quebras_nivel(y, n_efetivo=True)
        saida["deteccao"][serie] = (datas, tabela, datas_ef)
        sw, cs = teste_sup_wald(y), teste_cusum(y)
        lr_nb, lr_p = teste_razao_verossimilhanca(y, datas, "NB2"), teste_razao_verossimilhanca(y, datas, "Poisson")
        saida["globais"] += [
            {"série": serie, "teste": "sup-Wald de Andrews (HAC), 1 quebra de nível", "H0": "sem quebra de nível",
             "estatística": sw["sup-Wald"], "p": sw["p (assintótico de Andrews)"],
             "detalhe": f"data estimada {sw['data estimada']:%d/%m/%Y}; salto {sw['salto de nível estimado']:+.1f}/dia; VC 5% = {sw['valor crítico 5%']:.2f}"},
            {"série": serie, "teste": "OLS-CUSUM (resíduos de y ~ dia da semana)", "H0": "parâmetros estáveis",
             "estatística": cs["estatística OLS-CUSUM"], "p": cs["p"], "detalhe": "sensível a mudanças de média ao longo do tempo"},
            {"série": serie, "teste": f"Razão de verossimilhança NB2 ({len(datas)} dummies de regime)", "H0": "todas as dummies = 0",
             "estatística": lr_nb["LR"], "p": lr_nb["p"],
             "detalhe": "IRR: " + ", ".join(f"{k} = {v:.2f}" for k, v in lr_nb["IRR por regime"].items())},
            {"série": serie, "teste": f"Razão de verossimilhança Poisson ({len(datas)} dummies)", "H0": "todas as dummies = 0",
             "estatística": lr_p["LR"], "p": lr_p["p"], "detalhe": "referência; a NB2 acomoda a sobredispersão"},
        ]
        saida["locais"][serie] = confirmar_quebras_localmente(y, datas)
        saida[f"curva_sup_wald_{serie}"] = sw["curva"]
        for q in datas:
            saida["segmentadas"].append({"série": serie, **regressao_segmentada(y, q)})
    # Quebra da janela de vendas (sinal: proporção de vendas fora de 10h–19h)
    data_janela = detectar_mudanca_janela(transacoes, fim)
    fora = proporcao_fora_horario(transacoes, fim)
    sw_j = teste_sup_wald(fora.reindex(pd.date_range(fora.index.min(), fora.index.max())).fillna(0))
    saida["janela"] = {
        "data detectada (só treino)": data_janela,
        "data refinada (série inteira)": refinar_mudanca_janela(transacoes, data_janela) if data_janela is not None else None,
        "% fora 10–19h antes": 100 * fora.loc[:data_janela - pd.Timedelta(days=1)].mean(),
        "% fora 10–19h depois": 100 * fora.loc[data_janela:].mean(),
        "sup-Wald": sw_j["sup-Wald"], "p (Andrews)": sw_j["p (assintótico de Andrews)"],
        "data sup-Wald": sw_j["data estimada"],
        "Mann-Whitney p": stats.mannwhitneyu(fora.loc[:data_janela - pd.Timedelta(days=1)], fora.loc[data_janela:]).pvalue,
    }
    saida["globais"] = pd.DataFrame(saida["globais"])
    saida["segmentadas"] = pd.DataFrame(saida["segmentadas"])
    log.info("Quebras (até %s): %s | janela %s", f"{fim:%d/%m/%Y}",
             {s: [f"{d:%d/%m/%Y}" for d in v[0]] for s, v in saida["deteccao"].items()},
             f"{data_janela:%d/%m/%Y}" if data_janela is not None else "—")
    return saida


# ============================================================================ gráficos
def grafico_quebras(series: dict, resultado: dict, fim: pd.Timestamp) -> plt.Figure:
    """Demanda com médias por segmento (Binseg + BIC) e a curva do sup-Wald, para M1 e Total."""
    fig, eixos = plt.subplots(2, 2, figsize=(16, 7.5), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    for j, serie in enumerate(("Máquina 1", "Total")):
        y = series[(serie, "demanda")].loc[:fim]
        datas = resultado["deteccao"][serie][0]
        ax = eixos[0, j]
        ax.plot(y.index, y.values, color=CORES["contexto"], lw=0.8, label="Demanda diária")
        limites = [y.index.min()] + datas + [y.index.max() + pd.Timedelta(days=1)]
        for k in range(len(limites) - 1):
            trecho = y.loc[limites[k]: limites[k + 1] - pd.Timedelta(days=1)]
            ax.hlines(trecho.mean(), trecho.index.min(), trecho.index.max(), color=cor_serie(serie), lw=3,
                      label="Média do segmento" if k == 0 else None)
        reservar_topo(ax, 0.3)                       # rótulos das quebras e legenda acima dos picos diários
        anotar_quebras(ax, {f"{d:%d/%m/%y}": d for d in datas})
        ax.set_title(f"{serie}: segmentos de nível (Binseg + BIC, só treino)")
        ax.set_ylabel("transações/dia")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9))
        ax = eixos[1, j]
        curva = resultado[f"curva_sup_wald_{serie}"]
        vc = next(r for _, r in resultado["globais"].iterrows() if r["série"] == serie and r["teste"].startswith("sup-Wald"))
        ax.plot(curva.index, curva.values, color=cor_serie(serie), lw=1.6, label="Estatística de Wald (HAC) por data candidata")
        limite = float(vc["detalhe"].split("VC 5% = ")[1])
        ax.axhline(limite, color=CORES["quebra"], ls="--", lw=1.1, label=f"valor crítico 5% de Andrews ({limite:.1f})")
        ax.set_ylabel("Wald")
        ax.set_title("Teste sup-Wald: onde a evidência de quebra é máxima")
        ax.legend(loc="upper right", fontsize=8)     # canto vazio: a curva cai ao longo do tempo
        formatar_eixo_datas(ax)
    fig.tight_layout()
    return fig


def grafico_janela(transacoes: pd.DataFrame, data_janela: pd.Timestamp, data_refinada: pd.Timestamp) -> plt.Figure:
    """Primeira/última venda do dia e % de vendas fora de 10h–19h (Máquina 1)."""
    m1 = transacoes[transacoes["maquina"] == 1]
    jd = m1.groupby("date")["datetime"].agg(["min", "max"])
    fora = proporcao_fora_horario(transacoes)
    fig, eixos = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    ax = eixos[0]
    ax.scatter(jd.index, jd["min"].dt.hour + jd["min"].dt.minute / 60, s=9, color=CORES["Máquina 1"], label="Primeira venda do dia")
    ax.scatter(jd.index, jd["max"].dt.hour + jd["max"].dt.minute / 60, s=9, color=CORES["Máquina 2"], marker="D", label="Última venda do dia")
    ax.set_ylabel("hora do dia")
    ax.set_title("Máquina 1: janela de vendas inferida pela primeira e última venda do dia")
    reservar_topo(ax, 0.22)
    ax.legend(loc="upper right", ncol=2)
    ax = eixos[1]
    ax.plot(fora.index, 100 * fora.rolling(7, min_periods=3).mean(), color=CORES["Máquina 1"], lw=2)
    ax.set_ylabel("% fora de 10h–19h (média 7 dias)")
    ax.set_title("Proporção das vendas fora da janela original")
    reservar_topo(ax, 0.22)
    for a in eixos:
        anotar_quebras(a, {f"detectada só com treino: {data_janela:%d/%m/%Y}": data_janela})
        a.axvline(data_refinada, color=CORES["observado"], ls=":", lw=1)
    ax.annotate(f"1º dia com vendas fora da janela: {data_refinada:%d/%m/%Y}", xy=(data_refinada, 0.97),
                xycoords=("data", "axes fraction"), xytext=(-6, 0), textcoords="offset points", ha="right", va="top",
                fontsize=8.5, color=CORES["texto_secundario"])
    formatar_eixo_datas(ax)
    fig.tight_layout()
    return fig
