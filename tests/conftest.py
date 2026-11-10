"""Fixtures compartilhadas: base real (versão 21 do dataset, em data/raw) e detector de quebras."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest

from src.data.preprocessing import preparar_base
from src.structural_breaks.structural_break_test import DetectorQuebras

ORIGEM = pd.Timestamp("2025-03-09")


@pytest.fixture(scope="session")
def base():
    return preparar_base()


@pytest.fixture(scope="session")
def detector(base):
    return DetectorQuebras(base.series, base.transacoes)


@pytest.fixture(scope="session")
def origem():
    return ORIGEM
