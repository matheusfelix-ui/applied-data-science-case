"""Execução dos modelos: divisão cronológica, origem principal e backtesting em várias origens.

**Protocolo (sem embaralhamento).**

| Conjunto | Período | Uso |
|---|---|---|
| Treino | 01/03/2024 (M2: 08/02/2025) a 09/03/2025 | estimação |
| Validação interna | 3 dobras de 14 dias dentro do treino (origens 26/01, 09/02, 23/02) | escolha de janela, transformação, especificação e hiperparâmetros |
| Teste | 10/03 a 23/03/2025 (14 dias) | avaliação final; nunca usado no ajuste |
| Backtest | 10 origens anteriores a cada 14 dias + a principal = 11 blocos sem sobreposição (21/10/2024–23/03/2025) | estabilidade e significância (M1 e Total) |

Em cada origem do backtest **tudo é refeito só com os dados até ela** (quebras, janela de vendas,
especificações, hiperparâmetros, intervalos). As origens rodam em paralelo (``joblib``); os
resultados não dependem da ordem, pois todas as sementes são fixas.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data.preprocessing import BaseDados
from src.logging_config import obter_logger
from src.models import machine_learning_models as ml
from src.models.baselines import modelo_media_4_semanas, modelo_naive_sazonal
from src.models.model_selection import Contexto, NaoAplicavel, ResultadoModelo
from src.models.statistical_models import modelo_contagem, modelo_holt_winters, modelo_sarimax
from src.settings import carregar_config
from src.structural_breaks.structural_break_test import DetectorQuebras
from src.validation.metrics import calcular_metricas

log = obter_logger(__name__)
_CFG = carregar_config()

# Catálogo: nome → (família, função). A ordem é a de apresentação.
CATALOGO = {
    "Naive sazonal": ("Baseline", modelo_naive_sazonal),
    "Média 4 semanas": ("Baseline", modelo_media_4_semanas),
    "Holt-Winters": ("Estatístico — suavização exponencial", modelo_holt_winters),
    "SARIMAX": ("Estatístico — ARIMA com exógenas", modelo_sarimax),
    "Poisson/NB2": ("Estatístico — contagem", modelo_contagem),
    "Random Forest": ("Machine Learning", ml.modelo_random_forest),
}
BASELINES = [m for m, (f, _) in CATALOGO.items() if f == "Baseline"]


def datas_divisao(base: BaseDados) -> dict:
    """Datas da divisão cronológica, derivadas do fim da série e da configuração."""
    d = _CFG["divisao"]
    fim = base.fim
    inicio_teste = fim - pd.Timedelta(days=d["dias_teste"] - 1)
    origem = inicio_teste - pd.Timedelta(days=1)
    origens_bt = [origem - pd.Timedelta(days=d["horizonte"] * k) for k in range(d["n_origens_backtest"], 0, -1)]
    return {"fim": fim, "inicio_teste": inicio_teste, "origem": origem, "origens_backtest": origens_bt}


def executar_modelo(nome: str, ctx: Contexto) -> ResultadoModelo | dict:
    """Roda um modelo num contexto, valida a saída e aplica a restrição de não negatividade.

    Returns:
        ``ResultadoModelo`` ou, se o modelo não se aplica, ``{"motivo": ...}``.
    """
    familia, funcao = CATALOGO[nome]
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            saida = funcao(ctx)
    except NaoAplicavel as e:
        log.info("%s não aplicável a %s/%s (%s): %s", nome, ctx.serie, ctx.alvo, ctx.cenario, e)
        return {"modelo": nome, "serie": ctx.serie, "alvo": ctx.alvo, "cenario": ctx.cenario, "motivo": str(e)}
    previsao = pd.Series(np.clip(np.asarray(saida["previsao"], float), 0, None), index=ctx.datas_previsao)
    if len(previsao) != ctx.horizonte or previsao.isna().any():
        raise ValueError(f"{nome}/{ctx.serie}/{ctx.alvo}: previsão inválida")
    intervalos = saida.get("intervalos")
    if intervalos is not None:
        intervalos = pd.DataFrame({k: np.asarray(v, float) for k, v in dict(intervalos).items()},
                                  index=ctx.datas_previsao).clip(lower=0)
    ajustado = pd.Series(saida.get("ajustado", pd.Series(dtype=float))).reindex(ctx.y_treino.index).astype(float)
    extras = dict(saida.get("extras", {}))
    if "modelo" in saida:
        extras["objeto_modelo"] = saida["modelo"]
    return ResultadoModelo(nome, familia, ctx.serie, ctx.alvo, ctx.cenario, ctx.origem, previsao, ajustado, intervalos,
                           saida.get("especificacao", ""), saida.get("inicio_treino", ctx.y_treino.index.min()),
                           time.time() - t0, float(saida.get("mae_validacao", np.nan)), extras)


def executar_origem(base: BaseDados, origem: pd.Timestamp, series: list[tuple[str, str]], modelos: list[str] | None = None,
                    cenario: str | None = None, detector: DetectorQuebras | None = None, leve: bool = False) -> tuple[list, list]:
    """Executa os modelos em todas as séries de uma origem.

    Args:
        leve: descarta objetos pesados (modelos ajustados, previsões de validação) — usado no
            backtest paralelo, em que só as previsões interessam.
    """
    detector = detector or DetectorQuebras(base.series, base.transacoes)
    cenario = cenario or f"origem {origem:%d/%m/%Y}"
    resultados, nao_aplicaveis = [], []
    for serie, alvo in series:
        for nome in modelos or list(CATALOGO):
            r = executar_modelo(nome, Contexto(base, detector, serie, alvo, origem, cenario))
            if isinstance(r, dict):
                nao_aplicaveis.append(r)
                continue
            if leve:
                r.extras = {k: v for k, v in r.extras.items() if k in ("janela", "config")}
                r.ajustado = pd.Series(dtype=float)
            resultados.append(r)
    log.info("Origem %s concluída (%d resultados)", f"{origem:%d/%m/%Y}", len(resultados))
    return resultados, nao_aplicaveis


def _executar_origem_processo(base: BaseDados, origem: pd.Timestamp, series, modelos, regra_rf: str | None = None):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ml.definir_n_jobs(1)
    if regra_rf:
        ml.definir_regra(regra_rf)
    warnings.filterwarnings("ignore")
    return executar_origem(base, origem, series, modelos, leve=True)


def backtest(base: BaseDados, origens: list[pd.Timestamp], series: list[tuple[str, str]], modelos: list[str] | None = None,
             n_jobs: int | None = None, regra_rf: str | None = None) -> tuple[list, list]:
    """Executa todas as origens em paralelo (um processo por origem)."""
    n_jobs = n_jobs or min(len(origens), os.cpu_count() or 1)
    t0 = time.time()
    saidas = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_executar_origem_processo)(base, o, series, modelos, regra_rf) for o in origens)
    resultados = [r for s in saidas for r in s[0]]
    nao = [n for s in saidas for n in s[1]]
    log.info("Backtest: %d origens em %.0f s", len(origens), time.time() - t0)
    return resultados, nao


def tabela_resultados(resultados: list[ResultadoModelo], base: BaseDados) -> pd.DataFrame:
    """Métricas de cada resultado no período previsto (observado real), com o treino como escala."""
    linhas = []
    for r in resultados:
        s = base.series[(r.serie, r.alvo)]
        y = s.reindex(r.previsao.index)
        if y.isna().any():
            continue
        m = calcular_metricas(y, r.previsao, s.loc[:r.origem], r.intervalos)
        linhas.append({"modelo": r.modelo, "família": r.familia, "série": r.serie, "alvo": r.alvo, "origem": r.origem,
                       "cenário": r.cenario, **m, "MAE validação": r.mae_validacao, "especificação": r.especificacao,
                       "tempo_s": r.tempo_s})
    return pd.DataFrame(linhas)
