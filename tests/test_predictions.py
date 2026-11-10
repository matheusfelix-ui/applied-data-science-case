"""Testes das métricas, da comparação Total × soma, da seleção final e dos arquivos de previsão."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.model_selection import ResultadoModelo
from src.settings import RAIZ_PROJETO, carregar_config
from src.validation.metrics import calcular_metricas
from src.validation.model_comparison import comparar_bottom_up, diebold_mariano, selecao_final

CFG = carregar_config()


def test_metricas_calculadas_a_mao():
    y, p = np.array([10.0, 20.0, 30.0, 40.0]), np.array([12.0, 18.0, 33.0, 40.0])
    m = calcular_metricas(y, p)
    assert m["MAE"] == pytest.approx(7 / 4)
    assert m["RMSE"] == pytest.approx(np.sqrt((4 + 4 + 9 + 0) / 4))
    assert m["WAPE_%"] == pytest.approx(7 / 100 * 100)
    assert m["ME"] == pytest.approx(3 / 4)                     # > 0 = superestimação
    assert m["MAPE_%"] == pytest.approx(np.mean([0.2, 0.1, 0.1, 0]) * 100)


def test_mape_indefinido_com_zeros():
    assert np.isnan(calcular_metricas(np.array([0.0, 5.0]), np.array([1.0, 5.0]))["MAPE_%"])


def test_cobertura_e_winkler():
    y = np.array([5.0, 10.0])
    iv = pd.DataFrame({"li_80": [4, 4], "ls_80": [6, 6], "li_95": [3, 3], "ls_95": [7, 7]})
    m = calcular_metricas(y, np.array([5.0, 5.0]), intervalos=iv)
    assert m["cobertura_80_%"] == 50 and m["cobertura_95_%"] == 50
    assert m["winkler_95"] == pytest.approx(np.mean([4, 4 + 40 * 3]))


def test_diebold_mariano_sinal():
    rng = np.random.default_rng(0)
    e_bom, e_ruim = rng.normal(0, 1, 50), rng.normal(0, 3, 50)
    est, p = diebold_mariano(e_bom, e_ruim)
    assert est < 0 and p < 0.05


def _resultado(modelo, serie, alvo, valores, datas):
    return ResultadoModelo(modelo, "x", serie, alvo, "teste", datas[0] - pd.Timedelta(days=1), pd.Series(valores, index=datas),
                           pd.Series(dtype=float), None, "", datas[0], 0.0)


def test_soma_das_maquinas(base):
    datas = pd.date_range("2025-03-10", periods=14)
    r = [_resultado("M", "Máquina 1", "demanda", np.full(14, 10.0), datas),
         _resultado("M", "Máquina 2", "demanda", np.full(14, 5.0), datas),
         _resultado("M", "Total", "demanda", np.full(14, 16.0), datas)]
    t = comparar_bottom_up(r, base)
    y = base.series[("Total", "demanda")].loc[datas]
    assert t.loc[0, "MAE soma M1 + M2"] == pytest.approx(np.mean(np.abs(15.0 - y)))
    assert t.loc[0, "MAE Total direto"] == pytest.approx(np.mean(np.abs(16.0 - y)))


def test_selecao_final_respeita_filtro_de_vies():
    tab = pd.DataFrame({"série": "Máquina 2", "alvo": "demanda", "modelo": ["A", "B", "C"], "MAE": [1.0, 2.0, 3.0],
                        "WAPE_%": [10, 20, 30], "ME_relativo_%": [40.0, 5.0, -2.0], "MAE validação": [1.0, 2.0, 3.0]})
    sel = selecao_final(tab, None)
    escolhido = sel.loc[sel["escolhido"], "modelo"].iloc[0]
    assert escolhido == "B"          # A tem o menor erro, mas viés de 40% (> 15%) o desclassifica


# ------------------------------------------------------------- arquivos gerados pelo pipeline
PASTA = RAIZ_PROJETO / "reports" / "forecasts"
precisa_pipeline = pytest.mark.skipif(not (PASTA / "total" / "previsao_futura_14_dias.csv").exists(),
                                      reason="rode `python -m src.pipeline` para gerar as previsões")


@precisa_pipeline
@pytest.mark.parametrize("pasta", ["machine_1", "machine_2", "total"])
def test_arquivos_de_previsao(pasta):
    fut = pd.read_csv(PASTA / pasta / "previsao_futura_14_dias.csv", parse_dates=["data"])
    assert set(fut["alvo"]) == {"demanda", "receita"}
    for alvo, g in fut.groupby("alvo"):
        assert len(g) == CFG["divisao"]["horizonte_futuro"]
        assert g["data"].min() == pd.Timestamp("2025-03-24")
        assert (g["previsto"] >= 0).all()
        if "li_95" in g:
            assert (g["li_95"] <= g["previsto"] + 1e-6).all() and (g["previsto"] <= g["ls_95"] + 1e-6).all()
    teste = pd.read_csv(PASTA / pasta / "previsoes_teste_todos_modelos.csv")
    assert teste["modelo"].nunique() == 6
    assert (teste.groupby(["alvo", "modelo"]).size() == CFG["divisao"]["dias_teste"]).all()
    for ext in ("png", "pdf"):
        assert (PASTA / pasta / f"previsoes_modelo_escolhido.{ext}").exists()


@precisa_pipeline
def test_modelos_salvos_com_metadados():
    pasta = RAIZ_PROJETO / "models"
    registro = pd.read_csv(pasta / "registro_modelos.csv")
    assert len(registro) == 6
    for _, r in registro.iterrows():
        assert (pasta / f"{CFG['series']['pastas'][r['série']]}_{r['alvo']}_metadata.json").exists()


def test_r2_vale_1_na_previsao_perfeita_e_0_na_media_do_periodo():
    y = np.array([3.0, 7.0, 5.0, 9.0, 1.0])
    assert calcular_metricas(y, y)["R2"] == pytest.approx(1.0)
    assert calcular_metricas(y, np.full_like(y, y.mean()))["R2"] == pytest.approx(0.0)
    assert calcular_metricas(y, 2 * y.mean() - y)["R2"] == pytest.approx(-3.0)   # espelhada em torno da média
