"""Testes dos diagnósticos de tendência, sazonalidade, ciclo e estacionariedade (dados sintéticos)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eda import trend_seasonality as ts


def _serie(valores) -> pd.Series:
    return pd.Series(valores, index=pd.date_range("2024-03-04", periods=len(valores), freq="D"))


def test_fisher_pvalor_no_critico_e_monotono():
    n = 60
    g = ts.fisher_critico(n, 0.05)
    assert ts.fisher_pvalor(g, n) == pytest.approx(0.05, abs=1e-6)
    assert ts.fisher_pvalor(g * 0.8, n) > 0.05 > ts.fisher_pvalor(g * 1.2, n)


def test_fisher_tamanho_do_teste_em_ruido_branco():
    """Sob H0, a taxa de rejeição a 5% deve ficar perto de 5%."""
    rng = np.random.default_rng(2024)
    T, rep = 121, 2000
    g_crit = ts.fisher_critico((T - 1) // 2)
    rejeicoes = 0
    for _ in range(rep):
        _, I = ts.periodograma(rng.normal(size=T))
        rejeicoes += I.max() / I.sum() > g_crit
    assert 0.035 < rejeicoes / rep < 0.065


def test_periodo_fundamental_reconhece_harmonico():
    t = np.arange(140)
    x = np.cos(2 * np.pi * 2 * t / 7)           # só o 2º harmônico de 7 dias (pico em 3,5 dias)
    P, k = ts.periodo_fundamental(x, 2 / 7)
    assert (P, k) == (7, 2)


def test_metodologia_detecta_sazonalidade_semanal():
    rng = np.random.default_rng(1)
    t = np.arange(196)
    y = _serie(np.maximum(0, 20 + 0.02 * t + 6 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1.5, len(t))))
    r = ts.metodologia_fisher(y, "sintética", decidir_log=False)
    assert r.removidos[0] == 7
    assert r.tabela.loc[0, "p-valor"] < 0.001


def test_ruido_branco_sem_periodicidade():
    rng = np.random.default_rng(7)
    r = ts.metodologia_fisher(_serie(10 + rng.normal(size=210)), "ruído", decidir_log=False, grau_inicial=1)
    assert r.removidos == []


def test_degraus_vencem_tendencia_linear_quando_ha_mudanca_de_nivel():
    rng = np.random.default_rng(3)
    y = _serie(rng.poisson(np.r_[np.full(120, 6), np.full(120, 12), np.full(120, 8)]))
    quebras = [y.index[120], y.index[240]]
    tab = ts.comparar_formas_tendencia(y, quebras).set_index("modelo")
    assert tab.loc["degraus (quebras)", "ΔBIC"] == 0
    assert tab.loc["linear", "ΔBIC"] > 10


def test_diferenca_torna_passeio_aleatorio_estacionario():
    rng = np.random.default_rng(5)
    y = _serie(100 + np.cumsum(rng.normal(size=300)))
    tab = ts.testes_estacionariedade(y).set_index("transformação")
    assert tab.loc["original", "ADF p"] > 0.05
    assert tab.loc["Δ1 (1ª diferença)", "diagnóstico"] == "estacionária"


def test_sazonalidade_semanal_por_regime_e_teste_de_banda():
    rng = np.random.default_rng(11)
    t = np.arange(280)
    efeito = np.where(t >= 140, np.array([1.5, 1.4, 1.3, 1.2, 1.1, 0.5, 0.4])[pd.date_range("2024-03-04", periods=280).dayofweek], 1.0)
    y = _serie(rng.poisson(12 * efeito))
    tab, perfis, est = ts.teste_sazonalidade_semanal(y, [y.index[140]])
    assert tab.loc[0, "p"] > 0.01 and tab.loc[1, "p"] < 1e-6      # só o 2º regime tem padrão semanal
    assert est["p (último)"] < 1e-6
    assert perfis.shape == (2, 7)
    banda = ts.teste_fisher_banda(np.log1p(y.values[140:]))
    assert banda["período fundamental"] == 7 and banda["decisão"] == "rejeita H0"


def test_lambda_hp_tem_ganho_meio_no_corte():
    P = 13
    lamb = ts.lambda_hp_corte(P)
    w = 2 * np.pi / P
    ganho_suave = 1 / (1 + 4 * lamb * (1 - np.cos(w)) ** 2)
    assert ganho_suave == pytest.approx(0.5)


def test_serie_degraus_semanal_reproduz_nivel_por_regime():
    y = _serie(np.r_[np.full(70, 5.0), np.full(70, 15.0)])
    d = ts.serie_degraus_semanal(y, [y.index[70]])
    assert set(d.columns) == {"demanda", "nível do regime", "reta"}
    assert d["nível do regime"].iloc[0] == 5 and d["nível do regime"].iloc[-1] == 15
    assert d["reta"].is_monotonic_increasing


def test_fisher_recortes_detecta_periodo_7_so_no_trecho_com_semana():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-11-01", "2025-03-23", freq="D")
    efeito = np.where(idx >= "2025-02-01", np.where(idx.dayofweek >= 5, 0.4, 1.25), 1.0)
    y = pd.Series(rng.poisson(12 * efeito), index=idx).astype(float)
    tab = ts.fisher_recortes({("S", "demanda"): y}, [("S", "demanda")], {"2025": "2025-02-01", "antes": "2024-11-01"}, fins=(None, pd.Timestamp("2025-01-31")))
    so_2025 = tab[(tab["recorte"] == "2025")].iloc[0]
    assert so_2025["período 7 significativo"] and so_2025["p-valor"] < 0.01
    antes = tab[(tab["recorte"] == "antes") & (tab["fim"] == pd.Timestamp("2025-01-31"))].iloc[0]
    assert not antes["período 7 significativo"]
