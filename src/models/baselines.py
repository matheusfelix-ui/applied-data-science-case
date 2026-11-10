"""Modelos de referência (baselines): o mínimo que um modelo precisa superar para ter valor.

- **Naive sazonal semanal**: repete o mesmo dia da semana mais recente. Intervalos gaussianos com
  desvio σ·√(k + 1), k = semanas completas à frente (Hyndman & Athanasopoulos, FPP3, 5.5).
- **Média das últimas 4 semanas**: média das 4 últimas ocorrências do mesmo dia da semana. Mais
  estável que o naive, mas reage com atraso a mudanças de nível. É o adversário mais exigente nas
  avaliações anteriores (seção 8.5 de ``modelagens_testes``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.model_selection import Contexto, erro_validacao, intervalos_normais


def _naive(ctx: Contexto) -> np.ndarray:
    y, m = ctx.y_treino.to_numpy(), 7
    h = np.arange(1, ctx.horizonte + 1)
    return y[-m + (h - 1) % m]


def _media_4(ctx: Contexto) -> np.ndarray:
    y = ctx.y_treino
    saida = []
    for d in ctx.datas_previsao:
        # só ocorrências dentro do histórico disponível (a M2 começa em 08/02/2025)
        mesmos = [d - pd.Timedelta(days=7 * j) for j in range(1, 10)
                  if y.index.min() <= d - pd.Timedelta(days=7 * j) <= ctx.fim_treino]
        saida.append(y.loc[mesmos[:4]].mean())
    return np.asarray(saida)


def modelo_naive_sazonal(ctx: Contexto) -> dict:
    """ŷ(T + h) = y(T + h − 7k)."""
    y = ctx.y_treino
    previsao = _naive(ctx)
    ajustado = y.shift(7)
    sigma = (y - ajustado).dropna().std(ddof=1)
    h = np.arange(1, ctx.horizonte + 1)
    return {"previsao": previsao, "ajustado": ajustado,
            "intervalos": intervalos_normais(previsao, sigma * np.sqrt(np.floor((h - 1) / 7) + 1)),
            "especificacao": "ŷ(T+h) = y(T+h−7k)", "mae_validacao": erro_validacao(ctx, _naive)}


def modelo_media_4_semanas(ctx: Contexto) -> dict:
    """Média das 4 últimas ocorrências do mesmo dia da semana."""
    y = ctx.y_treino
    previsao = _media_4(ctx)
    ajustado = pd.concat([y.shift(7 * j) for j in range(1, 5)], axis=1).mean(axis=1, skipna=True)
    sigma = (y - ajustado).dropna().std(ddof=1)
    return {"previsao": previsao, "ajustado": ajustado,
            "intervalos": intervalos_normais(previsao, np.full(ctx.horizonte, sigma)),
            "especificacao": "média das 4 últimas ocorrências do mesmo dia da semana",
            "mae_validacao": erro_validacao(ctx, _media_4)}
