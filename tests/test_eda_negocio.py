"""Testes das análises de negócio da EDA: calendário das anomalias, concentração, ticket e padrão semanal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eda import duplicate_analysis as du, outlier_analysis as oa, product_analysis as pa, trend_seasonality as ts

FERIADOS = {"01-01": "Ano Novo", "05-01": "Dia do Trabalho", "12-25": "Natal"}


def test_classificar_dia_ordem_de_prioridade():
    assert oa.classificar_dia(pd.Timestamp("2025-01-01"), FERIADOS) == "feriado"
    assert oa.classificar_dia(pd.Timestamp("2024-12-26"), FERIADOS) == "dia seguinte a feriado"
    assert oa.classificar_dia(pd.Timestamp("2025-03-22"), FERIADOS) == "fim de semana"   # sábado
    assert oa.classificar_dia(pd.Timestamp("2025-03-19"), FERIADOS) == "dia útil"


def test_calendario_anomalias_cada_data_uma_vez_e_resumo_coerente(base):
    tabela, resumo = oa.calendario_anomalias(base)
    assert tabela["data"].is_unique
    assert set(resumo["sentido"]) <= {"queda", "pico"}
    assert resumo["dias"].sum() == len(tabela)
    # todos os dias sem venda da M1 entram como queda
    zeros = base.diarios["Máquina 1"].index[base.diarios["Máquina 1"]["demanda"] == 0]
    assert set(zeros) <= set(tabela.loc[tabela["sentido"] == "queda", "data"])


def test_curva_pareto_acumula_ate_100(base):
    for serie in ("Máquina 1", "Máquina 2"):
        p = pa.curva_pareto(base, serie)
        assert p["% acumulado"].is_monotonic_increasing
        assert abs(p["% acumulado"].iloc[-1] - 100) < 1e-9


def test_ticket_por_publico_compara_mesma_tabela_de_precos(base):
    """A leitura 'variação do ticket = efeito de mix' só vale se o preço de cada produto da M1 for o mesmo nos dois períodos."""
    _, mix, resumo = pa.ticket_por_publico(base, pd.Timestamp("2025-01-30"))
    m1 = base.transacoes[(base.transacoes["serie"] == "Máquina 1") & (base.transacoes["date"] >= resumo["última mudança de tabela da M1"])]
    assert m1.groupby("produto")["money"].nunique().max() == 1
    assert abs(mix["% antes"].sum() - 100) < 1e-6 and abs(mix["% depois"].sum() - 100) < 1e-6
    # ticket = Σ preço × participação (identidade contábil do efeito de mix)
    assert abs((mix["preço (u.m.)"] * mix["% depois"] / 100).sum() - resumo["ticket depois"]) < 1e-6


def test_perfil_semanal_detecta_padrao_so_depois_do_corte():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-03-04", periods=280, freq="D")
    corte = idx[224]
    lam = np.where(idx < corte, 10.0, 15.0 * np.where(idx.dayofweek >= 5, 0.45, 1.2))
    y = pd.Series(rng.poisson(lam), index=idx)
    perfis, testes = ts.perfil_semanal_antes_depois(y, [corte], corte)
    assert testes.loc[0, "p"] > 0.01 and testes.loc[1, "p"] < 0.001
    assert perfis.iloc[1][["sáb", "dom"]].max() < -30


def test_duplicatas_exatas_da_m2_estao_dentro_de_pedidos_no_mesmo_segundo(base):
    resumo = du.resumo_duplicados(base)
    assert resumo.loc["Máquina 1", "duplicatas exatas (registros a mais)"] == 0
    m2 = resumo.loc["Máquina 2"]
    assert m2["duplicatas exatas dentro desses grupos"] == m2["duplicatas exatas (registros a mais)"]
    assert m2["grupos com produtos diferentes"] >= 1   # o mesmo segundo também junta produtos diferentes: pedido com vários itens


def test_recompras_da_m1_nao_tem_intervalo_de_registro_em_dobro(base):
    rec, res = du.recompras_rapidas(base)
    assert res["recompras com intervalo < 30 s"] == 0
    assert (rec["intervalo (s)"] > 0).all()
    assert set(rec["tipo"]) == {"mesmo produto", "produto diferente"}
