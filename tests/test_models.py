"""Testes dos modelos: contrato de saída, exatidão dos baselines, transformações e reprodutibilidade."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.model_selection import (Contexto, intervalos_conformais, janelas_candidatas, media_original,
                                        quantil_original, transformar)
from src.validation.backtesting import CATALOGO, executar_modelo

RAPIDOS = ["Naive sazonal", "Média 4 semanas", "Holt-Winters", "SARIMAX", "Poisson/NB2"]


def _ctx(base, detector, origem, serie="Máquina 2", alvo="demanda"):
    return Contexto(base, detector, serie, alvo, origem)


def _checar_saida(r, ctx):
    assert len(r.previsao) == ctx.horizonte
    assert r.previsao.index.equals(ctx.datas_previsao)
    assert (r.previsao >= 0).all() and r.previsao.notna().all()
    assert r.previsao.index.min() > ctx.origem          # nada prevê o passado
    if r.intervalos is not None:
        iv = r.intervalos
        assert (iv["li_95"] <= iv["li_80"] + 1e-9).all() and (iv["li_80"] <= iv["ls_80"] + 1e-9).all()
        assert (iv["ls_80"] <= iv["ls_95"] + 1e-9).all() and (iv >= 0).all().all()


@pytest.mark.parametrize("modelo", RAPIDOS)
@pytest.mark.parametrize("alvo", ["demanda", "receita"])
def test_contrato_de_saida_maquina_2(base, detector, origem, modelo, alvo):
    ctx = _ctx(base, detector, origem, alvo=alvo)
    _checar_saida(executar_modelo(modelo, ctx), ctx)


@pytest.mark.slow
def test_contrato_random_forest(base, detector, origem):
    ctx = _ctx(base, detector, origem)
    r = executar_modelo("Random Forest", ctx)
    _checar_saida(r, ctx)
    assert "importancias" in r.extras


def test_catalogo_tem_2_baselines_e_4_candidatos():
    familias = [f for f, _ in CATALOGO.values()]
    assert familias.count("Baseline") == 2 and len(CATALOGO) == 6


def test_naive_sazonal_repete_a_ultima_semana(base, detector, origem):
    ctx = _ctx(base, detector, origem, "Máquina 1")
    r = executar_modelo("Naive sazonal", ctx)
    ultima = base.series[("Máquina 1", "demanda")].loc[:origem].iloc[-7:].to_numpy()
    assert np.allclose(r.previsao.to_numpy(), np.tile(ultima, 2))


def test_media_4_semanas(base, detector, origem):
    ctx = _ctx(base, detector, origem, "Máquina 1")
    r = executar_modelo("Média 4 semanas", ctx)
    y = base.series[("Máquina 1", "demanda")]
    d = ctx.datas_previsao[0]
    esperado = np.mean([y.loc[d - pd.Timedelta(days=7 * j)] for j in range(1, 5)])
    assert r.previsao.iloc[0] == pytest.approx(esperado)


def test_modelo_de_contagem_reprodutivel(base, detector, origem):
    a = executar_modelo("Poisson/NB2", _ctx(base, detector, origem))
    b = executar_modelo("Poisson/NB2", _ctx(base, detector, origem))
    assert np.allclose(a.previsao, b.previsao)


@pytest.mark.parametrize("alvo", ["demanda", "receita"])
def test_holt_winters_reprodutivel(base, detector, origem, alvo):
    """Com sazonalidade multiplicativa, o ETS calcula os intervalos por simulação: a semente tem de fixá-los."""
    a = executar_modelo("Holt-Winters", _ctx(base, detector, origem, serie="Máquina 1", alvo=alvo))
    b = executar_modelo("Holt-Winters", _ctx(base, detector, origem, serie="Máquina 1", alvo=alvo))
    assert np.array_equal(a.previsao.to_numpy(), b.previsao.to_numpy())
    assert a.intervalos.equals(b.intervalos)


@pytest.mark.parametrize("modelo", ["SARIMAX", "Média 4 semanas"])
def test_modelos_nao_veem_o_teste(base, detector, origem, modelo):
    """Alterar os valores do teste não muda a previsão (o contexto só contém o treino).

    Usa modelos com otimização determinística: o ETS tem oscilação numérica de ~0,03 entre execuções
    (otimizador), que não é vazamento; o modelo de contagem reaproveita um cache por origem.
    """
    ctx = _ctx(base, detector, origem, "Máquina 2")
    antes = executar_modelo(modelo, ctx).previsao
    serie = base.series[("Máquina 2", "demanda")]
    original = serie.loc[ctx.datas_previsao].copy()
    try:
        serie.loc[ctx.datas_previsao] = 999.0
        depois = executar_modelo(modelo, _ctx(base, detector, origem, "Máquina 2")).previsao
    finally:
        serie.loc[ctx.datas_previsao] = original
    assert np.allclose(antes, depois)


def test_transformacao_de_anscombe_inversivel():
    y = np.array([0, 1, 5, 20], dtype=float)
    z = transformar(y, "raiz (Anscombe)")
    assert np.allclose(quantil_original(z, "raiz (Anscombe)"), y)
    assert np.allclose(media_original(z, 0.0, "raiz (Anscombe)"), y)
    assert np.allclose(transformar(y, "nenhuma"), y)


def test_intervalos_conformais():
    iv = intervalos_conformais(np.full(5, 10.0), np.array([-1, 1, -2, 2, 3, -3, 1, 1, 2, 2]))
    assert (iv["ls_95"] >= iv["ls_80"]).all() and (iv["li_95"] >= 0).all()
    assert intervalos_conformais(np.ones(3), [1, 2]) is None     # poucos resíduos → sem intervalo


def test_janelas_candidatas_distintas(base, detector, origem):
    assert janelas_candidatas(_ctx(base, detector, origem, "Máquina 2")) == ["todo o treino"]
    assert len(janelas_candidatas(_ctx(base, detector, origem, "Máquina 1"))) == 4


def test_regra_um_erro_padrao_prefere_modelo_simples():
    from src.models import machine_learning_models as ml

    complexo = (10.0, dict(max_depth=None, min_samples_leaf=1, max_features=1.0, meia_vida=None, comportamentais=True), {}, 2.0, 1)
    simples = (11.0, dict(max_depth=8, min_samples_leaf=5, max_features=0.5, meia_vida=None, comportamentais=False), {}, 2.0, 1)
    longe = (20.0, dict(max_depth=8, min_samples_leaf=5, max_features=0.5, meia_vida=None, comportamentais=False), {}, 2.0, 1)
    assert ml._escolher_1ep([complexo, simples]) is simples        # dentro de 1 EP → o mais simples
    assert ml._escolher_1ep([complexo, longe]) is complexo         # fora de 1 EP → o melhor
