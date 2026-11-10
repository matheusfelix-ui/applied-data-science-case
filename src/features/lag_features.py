"""Defasagens e estatísticas de janela móvel (todas usam apenas valores até d − 1)."""
from __future__ import annotations

import pandas as pd

from src.settings import carregar_config


def atributos_defasados(y: pd.Series, lags: list[int] | None = None, janelas: list[int] | None = None,
                        prefixo: str = "") -> pd.DataFrame:
    """Defasagens e média/desvio/CV móveis para cada data de ``y.index``.

    Para o dia-alvo d, ``lag_k`` = y(d − k) e as janelas terminam em d − 1 (``shift(1)``), de modo
    que nenhum atributo usa o próprio valor do dia ou valores futuros. Dias sem histórico
    suficiente ficam NaN (tratado nativamente pelas árvores; os modelos estatísticos não os usam).
    """
    cfg = carregar_config()["features"]
    lags = lags or cfg["lags"]
    janelas = janelas or cfg["janelas_moveis"]
    X = pd.DataFrame(index=y.index)
    for k in lags:
        X[f"{prefixo}lag_{k}"] = y.shift(k)
    passado = y.shift(1)
    for w in janelas:
        media = passado.rolling(w, min_periods=w).mean()
        dp = passado.rolling(w, min_periods=w).std()
        X[f"{prefixo}media_{w}"] = media
        X[f"{prefixo}dp_{w}"] = dp
        X[f"{prefixo}cv_{w}"] = dp / media.where(media != 0)
    return X


def cauda_necessaria() -> int:
    """Nº mínimo de dias de histórico para recalcular todos os atributos de uma data (recursão)."""
    cfg = carregar_config()["features"]
    return max(cfg["lags"]) + max(cfg["janelas_moveis"]) + 5
