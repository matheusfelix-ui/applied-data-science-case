"""Modelos estatísticos candidatos: Holt-Winters (ETS), SARIMAX e modelo de contagem (Poisson/NB2).

Portados de ``modelagens_testes.ipynb`` (seções 6.1, 6.2–6.3 e 6.4), com a mesma lógica de seleção:

| Modelo | O que é escolhido na validação interna (MAE) | O que é escolhido por critério de informação |
|---|---|---|
| Holt-Winters | janela de estimação × transformação | especificação ETS sazonal (AICc) |
| SARIMAX | janela × transformação (com ordens por AICc) | ordens (auto_arima, AICc) e conjunto de exógenas (AICc) |
| Poisson/NB2 | janela de estimação | família × especificação (AIC) |

Na **receita**, o modelo de contagem não é reestimado: a receita de cada trajetória simulada é a
soma de N tickets sorteados dos últimos 28 dias (N = demanda simulada), um modelo de Poisson
composto. É uma previsão própria da receita, coerente com a demanda e com a distribuição de preços.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.features.temporal_features import dummies_dia_semana
from src.models.model_selection import (TRANSFORMACOES, Contexto, NaoAplicavel, erro_validacao, inicio_janela,
                                        janelas_candidatas, media_original, quantil_original, transformar)
from src.settings import carregar_config

_CFG = carregar_config()["modelos"]
SEMENTE = carregar_config()["projeto"]["semente"]


# ================================================================== Holt-Winters (ETS sazonal)
def _rotulo_ets(tendencia, sazonal) -> str:
    letra = {None: "N", "add": "A", "mul": "M"}
    return f"ETS(A,{letra[tendencia]}{'d' if tendencia else ''},{letra[sazonal]})"


def candidatos_ets(z: pd.Series, somente_sazonal: bool = True) -> pd.DataFrame:
    """ETS com erro aditivo; tendência nenhuma ou amortecida; sazonalidade aditiva ou multiplicativa (AICc)."""
    positivo = bool((z > 0).all())
    linhas = []
    for tendencia, sazonal in itertools.product([None, "add"], [None, "add", "mul"]):
        if (somente_sazonal and sazonal is None) or (sazonal == "mul" and not positivo):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ETSModel(z, error="add", trend=tendencia, damped_trend=tendencia is not None, seasonal=sazonal,
                               seasonal_periods=7 if sazonal else None).fit(disp=False, maxiter=3000)
        except Exception:
            continue
        linhas.append({"especificação": _rotulo_ets(tendencia, sazonal), "sazonal": sazonal is not None,
                       "AICc": res.aicc, "res": res})
    if not linhas:
        raise NaoAplicavel("nenhuma especificação ETS estimável")
    return pd.DataFrame(linhas).sort_values("AICc").reset_index(drop=True)


def _ets_janela(ctx: Contexto, janela: str, transformacao: str, completo: bool = False):
    inicio = inicio_janela(ctx, janela)
    y = ctx.y_treino.loc[inicio:]
    z = pd.Series(transformar(y, transformacao), index=y.index)
    tabela = candidatos_ets(z)
    res = tabela.loc[0, "res"]
    # Com componente multiplicativo, o statsmodels calcula os intervalos por simulação: sem semente, eles (e a média
    # na escala original, que usa a variância) mudariam a cada execução. O gerador é recriado a cada chamada.
    pred = res.get_prediction(start=ctx.datas_previsao[0], end=ctx.datas_previsao[-1], rng=np.random.default_rng(SEMENTE))
    q95 = pred.summary_frame(alpha=0.05)
    var = ((q95["pi_upper"] - q95["mean"]) / stats.norm.ppf(0.975)).to_numpy() ** 2
    previsao = media_original(q95["mean"].to_numpy(), var, transformacao)
    if not completo:
        return previsao
    intervalos = {}
    for nivel in (80, 95):
        q = pred.summary_frame(alpha=1 - nivel / 100)
        intervalos[f"li_{nivel}"] = quantil_original(q["pi_lower"].to_numpy(), transformacao)
        intervalos[f"ls_{nivel}"] = quantil_original(q["pi_upper"].to_numpy(), transformacao)
    ajustado = pd.Series(media_original(res.fittedvalues.to_numpy(), float(np.var(res.resid)), transformacao), index=y.index)
    return {"previsao": previsao, "ajustado": ajustado, "intervalos": pd.DataFrame(intervalos), "inicio_treino": inicio,
            "especificacao_ets": tabela.loc[0, "especificação"], "modelo": res, "residuos": pd.Series(res.resid, index=y.index)}


def modelo_holt_winters(ctx: Contexto) -> dict:
    """Holt-Winters: janela × transformação pelo MAE de validação; ETS sazonal por AICc dentro de cada uma."""
    linhas = []
    for janela, transf in itertools.product(janelas_candidatas(ctx), TRANSFORMACOES):
        mae = erro_validacao(ctx, lambda sub, j=janela, t=transf: _ets_janela(sub, j, t))
        linhas.append({"janela": janela, "transformação": transf, "MAE validação": mae})
    escolha = pd.DataFrame(linhas).sort_values("MAE validação", na_position="last").reset_index(drop=True)
    janela, transf = escolha.loc[0, "janela"], escolha.loc[0, "transformação"]
    saida = _ets_janela(ctx, janela, transf, completo=True)
    return {**saida, "especificacao": f"{saida.pop('especificacao_ets')} · {janela} · {transf}",
            "mae_validacao": escolha.loc[0, "MAE validação"],
            "extras": {"candidatos": escolha, "janela": janela, "transformação": transf}}


# ================================================================== SARIMAX
def _auto_arima(y) -> pm.arima.ARIMA:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pm.auto_arima(np.asarray(y, float), seasonal=True, m=7, information_criterion="aicc", stepwise=True,
                             max_p=_CFG["sarima"]["max_p"], max_q=_CFG["sarima"]["max_q"], max_P=_CFG["sarima"]["max_P"],
                             max_Q=_CFG["sarima"]["max_Q"], max_d=1, max_D=1, test="kpss", seasonal_test="ocsb",
                             suppress_warnings=True, error_action="ignore", trace=False)


def _rotulo_arima(m) -> str:
    p, d, q = m.order
    P, D, Q, s = m.seasonal_order
    return f"SARIMA({p},{d},{q})({P},{D},{Q})[{s}]" if s > 1 else f"ARIMA({p},{d},{q})"


def _prever_arima(m, transf: str, h: int) -> np.ndarray:
    media_z, ic95 = m.predict(n_periods=h, return_conf_int=True, alpha=0.05)
    var = ((ic95[:, 1] - media_z) / stats.norm.ppf(0.975)) ** 2
    return media_original(media_z, var, transf)


def _reestimar_arima(m, y_z):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pm.ARIMA(order=m.order, seasonal_order=m.seasonal_order, with_intercept=m.with_intercept,
                        suppress_warnings=True).fit(np.asarray(y_z, float))


def _ajustar_sarimax(z: pd.Series, X: pd.DataFrame | None, ordens, tendencia: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(z, exog=X, order=ordens.order, seasonal_order=ordens.seasonal_order, trend=tendencia).fit(
            disp=False, maxiter=500)


def modelo_sarimax(ctx: Contexto) -> dict:
    """Regressão com erros SARIMA e exógenas determinísticas (calendário e dummies de quebra).

    1. Para cada janela × transformação, ordens por AICc (auto_arima) e MAE nas dobras internas
       (mesmas ordens reestimadas até cada origem) → escolhe janela e transformação.
    2. Conjuntos aninhados de exógenas comparados por AICc (mesma série, escala e ordens):
       calendário; calendário + quebras (``dummy_horario_reduzido``, ``dummy_novos_clientes``,
       ``regime_*``); completo (+ início do mês). No Total, ``maquinas_ativas`` entra sempre.
       Colunas constantes na janela são descartadas automaticamente. Como em ``modelagens_testes``,
       o SARIMAX sempre leva ao menos uma exógena; o ΔAICc contra o SARIMA sem exógenas é reportado
       (positivo = as exógenas não melhoram o ajuste).
    """
    linhas, modelos = [], {}
    for janela, transf in itertools.product(janelas_candidatas(ctx), TRANSFORMACOES):
        y = ctx.y_treino.loc[inicio_janela(ctx, janela):]
        m = _auto_arima(transformar(y, transf))
        modelos[(janela, transf)] = m

        def prever(sub, j=janela, t=transf, m=m):
            return _prever_arima(_reestimar_arima(m, transformar(sub.y_treino.loc[inicio_janela(sub, j):], t)), t, sub.horizonte)

        linhas.append({"janela": janela, "transformação": transf, "ordens (AICc)": _rotulo_arima(m),
                       "MAE validação (sem exógenas)": erro_validacao(ctx, prever)})
    escolha = pd.DataFrame(linhas).sort_values("MAE validação (sem exógenas)", na_position="last").reset_index(drop=True)
    janela, transf = escolha.loc[0, "janela"], escolha.loc[0, "transformação"]
    ordens = modelos[(janela, transf)]
    inicio = inicio_janela(ctx, janela)
    y = ctx.y_treino.loc[inicio:]
    z = pd.Series(transformar(y, transf), index=y.index)
    X_treino, X_fut = ctx.exogenas(y.index), ctx.exogenas(ctx.datas_previsao)
    quebras = [c for c in X_treino if c.startswith(("dummy_", "regime_"))]
    fixas = ["maquinas_ativas"] if ctx.serie == "Total" else []
    conjuntos = {"calendário": ["feriado", "data_comercial"], "calendário + quebras": ["feriado", "data_comercial"] + quebras,
                 "completo": ["feriado", "data_comercial", "inicio_mes"] + quebras}
    tendencia = "c" if ordens.with_intercept else "n"
    sem_exog = _ajustar_sarimax(z, None, ordens, tendencia)
    cand, vistos = [], set()
    for nome, cols in conjuntos.items():
        cols = [c for c in fixas + cols if c in X_treino and X_treino[c].nunique() > 1]
        if not cols or tuple(cols) in vistos:
            continue
        vistos.add(tuple(cols))
        try:
            res = _ajustar_sarimax(z, X_treino[cols], ordens, tendencia)
        except Exception:
            continue
        cand.append({"conjunto": nome, "variáveis": ", ".join(cols), "AICc": res.aicc,
                     "ΔAICc vs sem exógenas": res.aicc - sem_exog.aicc, "res": res, "colunas": cols})
    if not cand:     # nenhuma exógena varia na janela: SARIMAX degenera em SARIMA (registrado na especificação)
        cand.append({"conjunto": "sem exógenas", "variáveis": "—", "AICc": sem_exog.aicc, "ΔAICc vs sem exógenas": 0.0,
                     "res": sem_exog, "colunas": []})
    tabela = pd.DataFrame(cand).sort_values("AICc").reset_index(drop=True)
    melhor, cols = tabela.loc[0, "res"], tabela.loc[0, "colunas"]
    prev = melhor.get_forecast(steps=ctx.horizonte, exog=X_fut[cols] if cols else None)
    var = np.asarray(prev.var_pred_mean)
    intervalos = {}
    for nivel in (80, 95):
        ic = prev.conf_int(alpha=1 - nivel / 100).to_numpy()
        intervalos[f"li_{nivel}"] = quantil_original(ic[:, 0], transf)
        intervalos[f"ls_{nivel}"] = quantil_original(ic[:, 1], transf)
    coef = pd.DataFrame({"coeficiente": melhor.params[cols], "erro-padrão": melhor.bse[cols], "p-valor": melhor.pvalues[cols]}) if cols else pd.DataFrame()

    def prever_final(sub):
        ys = sub.y_treino.loc[inicio_janela(sub, janela):]
        Xs, Xf = sub.exogenas(ys.index), sub.exogenas(sub.datas_previsao)
        cs = [c for c in cols if c in Xs and Xs[c].nunique() > 1]
        r = _ajustar_sarimax(pd.Series(transformar(ys, transf), index=ys.index), Xs[cs] if cs else None, ordens, tendencia)
        p = r.get_forecast(steps=sub.horizonte, exog=Xf[cs] if cs else None)
        return media_original(p.predicted_mean.to_numpy(), np.asarray(p.var_pred_mean), transf)

    residuos = pd.Series(np.asarray(melhor.resid), index=y.index).iloc[14:]
    ljung = acorr_ljungbox(residuos, lags=[min(14, max(1, len(residuos) // 3))])["lb_pvalue"].iloc[0]
    ajustado = media_original(np.asarray(melhor.fittedvalues), float(np.var(residuos)), transf)
    return {"previsao": media_original(prev.predicted_mean.to_numpy(), var, transf),
            "ajustado": pd.Series(ajustado, index=y.index), "intervalos": pd.DataFrame(intervalos), "inicio_treino": inicio,
            "especificacao": f"{_rotulo_arima(ordens)} + [{tabela.loc[0, 'variáveis']}] · {janela} · {transf}",
            "mae_validacao": erro_validacao(ctx, prever_final), "modelo": melhor,
            "extras": {"candidatos": escolha, "exogenas": tabela.drop(columns=["res", "colunas"]), "coeficientes": coef,
                       "Ljung-Box p": ljung, "janela": janela, "transformação": transf, "residuos": residuos}}


# ================================================================== Contagem (Poisson / NB2)
ESPECIFICACOES = {"nível + exógenas": dict(semana=False, dinamica=False),
                  "+ dia da semana": dict(semana=True, dinamica=False),
                  "+ dinâmica": dict(semana=False, dinamica=True),
                  "+ dia da semana + dinâmica": dict(semana=True, dinamica=True)}
DINAMICA = ["log_lag_1", "log_lag_7", "log_media_7"]


def _exog_contagem(ctx: Contexto, datas) -> pd.DataFrame:
    X = ctx.exogenas(datas)
    return X[["feriado"] + [c for c in X if c.startswith(("dummy_", "regime_")) or c == "maquinas_ativas"]]


def _matriz_contagem(ctx: Contexto, datas, espec: dict, colunas_exog: list[str]) -> pd.DataFrame:
    y = ctx.y_treino
    partes = [pd.DataFrame({"const": 1.0}, index=datas)]
    if espec["semana"]:
        partes.append(dummies_dia_semana(datas))
    partes.append(_exog_contagem(ctx, datas)[colunas_exog])
    if espec["dinamica"]:
        partes.append(pd.DataFrame({"log_lag_1": np.log1p(y.shift(1)), "log_lag_7": np.log1p(y.shift(7)),
                                    "log_media_7": np.log1p(y.shift(1).rolling(7).mean())}).reindex(datas))
    return pd.concat(partes, axis=1)


def _ajustar_especificacoes(ctx: Contexto, janela: str):
    inicio = max(inicio_janela(ctx, janela), ctx.y_treino.index.min() + pd.Timedelta(days=7))
    datas = ctx.y_treino.loc[inicio:].index
    y = ctx.y_treino.loc[datas]
    exog = _exog_contagem(ctx, datas)
    colunas_exog = [c for c in exog if exog[c].nunique() > 1]
    linhas = []
    for (nome, espec), familia in itertools.product(ESPECIFICACOES.items(), ["Poisson", "NB2"]):
        X = _matriz_contagem(ctx, datas, espec, colunas_exog)
        if len(y) <= X.shape[1] + 5:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if familia == "Poisson":
                    res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
                else:
                    res = NegativeBinomial(y, X, loglike_method="nb2").fit(disp=0, method="bfgs", maxiter=1000)
                    # α na fronteira (≈ 0) ou sem convergência: a NB2 coincide com a Poisson e não é identificada
                    if not res.mle_retvals.get("converged", True) or float(res.params["alpha"]) < 1e-3:
                        continue
        except Exception:
            continue
        if np.isfinite(res.llf):
            linhas.append({"especificação": nome, "família": familia, "AIC": res.aic, "res": res, "espec": espec})
    if not linhas:
        raise NaoAplicavel("nenhuma especificação de contagem estimável")
    return pd.DataFrame(linhas).sort_values("AIC").reset_index(drop=True), datas, colunas_exog


def _simular(ctx: Contexto, res, espec, colunas_exog, n, rng) -> np.ndarray:
    """Trajetórias futuras: cada passo sorteia da distribuição e realimenta as defasagens da trajetória."""
    p = res.params
    alfa = float(p["alpha"]) if "alpha" in p.index else 0.0
    beta = p.drop("alpha", errors="ignore")
    X_fixo = _matriz_contagem(ctx, ctx.datas_previsao, {**espec, "dinamica": False}, colunas_exog)
    eta_fixo = X_fixo.to_numpy() @ beta[X_fixo.columns].to_numpy()
    caminhos = np.tile(ctx.y_treino.to_numpy()[-7:], (n, 1)).astype(float)
    saida = np.empty((n, ctx.horizonte))
    for h in range(ctx.horizonte):
        eta = np.full(n, eta_fixo[h])
        if espec["dinamica"]:
            din = np.column_stack([np.log1p(caminhos[:, -1]), np.log1p(caminhos[:, -7]), np.log1p(caminhos[:, -7:].mean(axis=1))])
            eta = eta + din @ beta[DINAMICA].to_numpy()
        mu = np.exp(np.clip(eta, -20, 20))
        sorteio = rng.negative_binomial(1 / alfa, 1 / (1 + alfa * mu)) if alfa > 0 else rng.poisson(mu)
        saida[:, h] = sorteio
        caminhos = np.column_stack([caminhos[:, 1:], sorteio])
    return saida


def _contagem_janela(ctx: Contexto, janela: str, n: int):
    tabela, datas, colunas_exog = _ajustar_especificacoes(ctx, janela)
    m = tabela.loc[0]
    return _simular(ctx, m["res"], m["espec"], colunas_exog, n, np.random.default_rng(SEMENTE)), tabela, datas, colunas_exog


def _ajustar_demanda(ctx: Contexto) -> dict:
    linhas = []
    for janela in janelas_candidatas(ctx):
        mae = erro_validacao(ctx, lambda sub, j=janela: _contagem_janela(sub, j, _CFG["contagem"]["simulacoes_validacao"])[0].mean(axis=0))
        linhas.append({"janela": janela, "MAE validação (demanda)": mae})
    escolha = pd.DataFrame(linhas).sort_values("MAE validação (demanda)", na_position="last").reset_index(drop=True)
    janela = escolha.loc[0, "janela"]
    traj, tabela, datas, colunas_exog = _contagem_janela(ctx, janela, _CFG["contagem"]["simulacoes_final"])
    res, espec = tabela.loc[0, "res"], tabela.loc[0, "espec"]
    X = _matriz_contagem(ctx, datas, espec, colunas_exog)
    y = ctx.y_treino.loc[datas]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        poisson = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        try:
            nb = NegativeBinomial(y, X, loglike_method="nb2").fit(disp=0, method="bfgs", maxiter=1000)
            alfa, razao = float(nb.params["alpha"]), 2 * (nb.llf - poisson.llf)
        except Exception:
            alfa, razao = np.nan, np.nan
    return {"trajetorias": traj, "ajustado": pd.Series(np.asarray(res.predict(X)), index=datas), "inicio": datas.min(),
            "janela": janela, "especificacao": f"{tabela.loc[0, 'família']} [{tabela.loc[0, 'especificação']}]",
            "escolha": escolha, "tabela": tabela.drop(columns=["res", "espec"]), "modelo": res,
            "coeficientes": pd.DataFrame({"coeficiente": res.params, "erro-padrão": res.bse, "p-valor": res.pvalues}),
            "α (NB2)": alfa, "p (α = 0)": 0.5 * stats.chi2.sf(max(razao, 0), 1) if np.isfinite(razao) else np.nan,
            "mae_validacao": escolha.loc[0, "MAE validação (demanda)"]}


def _tickets(ctx: Contexto, inicio: pd.Timestamp, fim: pd.Timestamp) -> np.ndarray:
    t = ctx.base.transacoes
    if ctx.serie != "Total":
        t = t[t["serie"] == ctx.serie]
    return t.loc[t["date"].between(inicio, fim), "money"].to_numpy(float)


def _receita_composta(trajetorias: np.ndarray, tickets: np.ndarray, rng) -> np.ndarray:
    sorteio = rng.choice(tickets, size=int(trajetorias.sum()))
    celula = np.repeat(np.arange(trajetorias.size), trajetorias.astype(int).ravel())
    return np.bincount(celula, weights=sorteio, minlength=trajetorias.size).reshape(trajetorias.shape)


_CACHE_DEMANDA: dict = {}


def modelo_contagem(ctx: Contexto) -> dict:
    """Poisson/NB2 para a demanda; Poisson composto (demanda × tickets sorteados) para a receita."""
    chave = (ctx.serie, ctx.origem)
    if chave not in _CACHE_DEMANDA:      # a demanda é ajustada uma vez e reaproveitada pela receita
        _CACHE_DEMANDA[chave] = _ajustar_demanda(ctx if ctx.alvo == "demanda" else ctx.sub(ctx.origem, alvo="demanda"))
    base = _CACHE_DEMANDA[chave]
    traj, ajustado = base["trajetorias"], base["ajustado"]
    rotulo = f"{base['especificacao']} · {base['janela']}"
    mae_val = base["mae_validacao"]
    if ctx.alvo == "receita":
        dias = _CFG["contagem"]["dias_tickets"]
        tickets = _tickets(ctx, ctx.origem - pd.Timedelta(days=dias - 1), ctx.origem)
        if len(tickets) == 0:
            raise NaoAplicavel("sem tickets recentes para compor a receita")
        traj = _receita_composta(traj, tickets, np.random.default_rng(SEMENTE))
        rec, dem = ctx.base.series[(ctx.serie, "receita")], ctx.base.series[(ctx.serie, "demanda")]
        ticket_movel = rec.shift(1).rolling(dias, min_periods=7).sum() / dem.shift(1).rolling(dias, min_periods=7).sum().replace(0, np.nan)
        ajustado = ajustado * ticket_movel.reindex(ajustado.index)
        rotulo += f" · receita = demanda × tickets ({dias} dias)"

        def prever_rec(sub):
            d = _contagem_janela(sub.sub(sub.origem, "demanda"), base["janela"], _CFG["contagem"]["simulacoes_validacao"])[0]
            tk = _tickets(sub, sub.origem - pd.Timedelta(days=dias - 1), sub.origem)
            return _receita_composta(d, tk, np.random.default_rng(SEMENTE)).mean(axis=0)

        mae_val = erro_validacao(ctx, prever_rec)
    intervalos = pd.DataFrame({f"{lado}_{nivel}": np.quantile(traj, q, axis=0)
                               for nivel in (80, 95) for lado, q in [("li", 0.5 - nivel / 200), ("ls", 0.5 + nivel / 200)]})
    return {"previsao": traj.mean(axis=0), "ajustado": ajustado, "intervalos": intervalos, "inicio_treino": base["inicio"],
            "especificacao": rotulo, "mae_validacao": mae_val, "modelo": base["modelo"],
            "extras": {k: base[k] for k in ["escolha", "tabela", "coeficientes", "α (NB2)", "p (α = 0)", "janela"]}}
