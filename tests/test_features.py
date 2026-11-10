"""Testes de atributos: ausência de vazamento, consistência da recursão e datas das dummies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.break_dummies import dummies_quebras
from src.features.feature_pipeline import ConstrutorAtributos, teste_consistencia_recursao, teste_vazamento_futuro
from src.features.lag_features import atributos_defasados
from src.features.temporal_features import features_calendario


@pytest.mark.parametrize("serie", ["Máquina 1", "Máquina 2", "Total"])
def test_sem_vazamento_do_futuro(base, detector, origem, serie):
    c = ConstrutorAtributos(base, detector, serie, origem)
    assert teste_vazamento_futuro(c, base.series[(serie, "demanda")])


@pytest.mark.parametrize("serie", ["Máquina 1", "Total"])
def test_recursao_consistente(base, detector, origem, serie):
    c = ConstrutorAtributos(base, detector, serie, origem)
    assert teste_consistencia_recursao(c, base.series[(serie, "demanda")].loc[:origem], n_datas=10)


def test_defasagens_usam_somente_o_passado():
    y = pd.Series(np.arange(40, dtype=float), index=pd.date_range("2025-01-01", periods=40))
    X = atributos_defasados(y, lags=[1, 7], janelas=[7])
    d = y.index[20]
    assert X.loc[d, "lag_1"] == y.iloc[19]
    assert X.loc[d, "lag_7"] == y.iloc[13]
    assert X.loc[d, "media_7"] == y.iloc[13:20].mean()   # janela termina em d − 1


def test_calendario():
    datas = pd.date_range("2025-03-08", periods=3)   # sáb, dom, seg
    X = features_calendario(datas)
    assert X["fim_de_semana"].tolist() == [1, 1, 0]
    assert X["data_comercial"].tolist() == [1, 0, 0]  # 08/03
    assert X.filter(like="fourier").abs().le(1).all().all()


def test_datas_das_dummies_na_origem_principal(base, detector, origem):
    datas = base.series[("Máquina 1", "demanda")].index
    D = dummies_quebras(datas, "Máquina 1", origem, detector, base.inicio["Máquina 2"])
    # período de horário reduzido termina na véspera da extensão detectada só com o treino (18/05/2024)
    assert D.loc["2024-05-17", "dummy_horario_reduzido"] == 1
    assert D.loc["2024-05-18", "dummy_horario_reduzido"] == 0
    # novos clientes a partir da quebra de 2025 (30/01 na M1)
    assert D.loc["2025-01-29", "dummy_novos_clientes"] == 0
    assert D.loc["2025-01-30", "dummy_novos_clientes"] == 1
    Dt = dummies_quebras(datas, "Total", origem, detector, base.inicio["Máquina 2"])
    assert Dt.loc["2025-02-04", "dummy_novos_clientes"] == 1 and Dt.loc["2025-02-03", "dummy_novos_clientes"] == 0
    assert Dt.loc["2025-02-08", "maquinas_ativas"] == 2 and Dt.loc["2025-02-07", "maquinas_ativas"] == 1


def test_dummy_nao_existe_antes_de_a_quebra_ser_detectavel(base, detector):
    """Numa origem de out/2024, a quebra de 2025 não pode aparecer (seria informação do futuro)."""
    origem_antiga = pd.Timestamp("2024-10-20")
    datas = pd.date_range("2024-10-21", periods=14)
    D = dummies_quebras(datas, "Máquina 1", origem_antiga, detector, base.inicio["Máquina 2"])
    assert (D["dummy_novos_clientes"] == 0).all()


def test_dummies_futuras_mantem_regime_vigente(base, detector, origem):
    futuro = pd.date_range(origem + pd.Timedelta(days=1), periods=14)
    D = dummies_quebras(futuro, "Máquina 1", origem, detector, base.inicio["Máquina 2"])
    assert (D["dummy_novos_clientes"] == 1).all() and (D["dummy_horario_reduzido"] == 0).all()


def test_comportamentais_congelados_no_horizonte(base, detector, origem):
    c = ConstrutorAtributos(base, detector, "Máquina 1", origem)
    y = base.series[("Máquina 1", "demanda")].loc[:origem].copy()
    for d in pd.date_range(origem + pd.Timedelta(days=1), periods=5):
        y.loc[d] = np.nan
    X = c(y).filter(like="comp_")
    futuras = X.loc[origem + pd.Timedelta(days=1):]
    assert (futuras.nunique() <= 1).all()
