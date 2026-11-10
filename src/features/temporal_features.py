"""Variáveis de calendário: conhecidas de antemão para qualquer data futura (sem risco de vazamento)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.settings import carregar_config


def features_calendario(datas: pd.DatetimeIndex, fourier_ordem: int | None = None) -> pd.DataFrame:
    """Calendário e sazonalidade semanal para as datas informadas.

    Colunas: dia da semana, dia do mês, mês, semana do ano, fim de semana, feriado, data comercial,
    início do mês (dias 1–5) e termos de Fourier semanais (sen/cos de ordem 1..K).

    O país não é informado pela base; feriados e datas comerciais são uma lista genérica
    (``config.yaml``). Nenhuma dessas datas cai no período de teste.
    """
    cfg = carregar_config()
    k_max = fourier_ordem or cfg["features"]["fourier_ordem"]
    mmdd = datas.strftime("%m-%d")
    X = pd.DataFrame(index=datas)
    X["dia_semana"] = datas.dayofweek
    X["dia_mes"] = datas.day
    X["mes"] = datas.month
    X["semana_ano"] = datas.isocalendar().week.astype(int).to_numpy()
    X["fim_de_semana"] = (datas.dayofweek >= 5).astype(int)
    X["feriado"] = mmdd.isin(list(cfg["calendario"]["feriados"])).astype(int)
    X["data_comercial"] = mmdd.isin(list(cfg["calendario"]["datas_comerciais"])).astype(int)
    X["inicio_mes"] = (datas.day <= 5).astype(int)
    for k in range(1, k_max + 1):
        X[f"fourier_sen_{k}"] = np.sin(2 * np.pi * k * datas.dayofweek / 7)
        X[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * datas.dayofweek / 7)
    return X


def dummies_dia_semana(datas: pd.DatetimeIndex) -> pd.DataFrame:
    """Seis dummies de dia da semana (segunda-feira é a referência)."""
    nomes = ["ter", "qua", "qui", "sex", "sab", "dom"]
    return pd.DataFrame({f"dow_{nomes[k - 1]}": (datas.dayofweek == k).astype(int) for k in range(1, 7)}, index=datas)
