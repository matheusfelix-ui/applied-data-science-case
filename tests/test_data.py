"""Testes da camada de dados: contrato dos arquivos brutos, semântica das séries e invariantes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import ErroValidacaoDados, validar_bruto
from src.data.preprocessing import agregar_diario
from src.settings import carregar_config

CFG = carregar_config()


def _bruto_valido(n=3):
    return pd.DataFrame({"date": ["2025-01-01"] * n, "datetime": ["2025-01-01 10:00:00"] * n,
                         "cash_type": ["card"] * n, "money": [30.0] * n, "coffee_name": ["Latte"] * n})


def test_validacao_aceita_arquivo_valido():
    validar_bruto(_bruto_valido(), "ok.csv", CFG)


@pytest.mark.parametrize("alteracao, mensagem", [
    (lambda d: d.drop(columns="money"), "colunas ausentes"),
    (lambda d: d.assign(money=[-1.0, 30.0, 30.0]), "não positivos"),
    (lambda d: d.assign(money=["abc", 30.0, 30.0]), "não numéricos"),
    (lambda d: d.assign(cash_type=["pix", "card", "card"]), "fora do domínio"),
    (lambda d: d.assign(datetime=["ontem", "2025-01-01 10:00:00", "2025-01-01 10:00:00"]), "datetime inválidos"),
    (lambda d: d.assign(date=["2025-01-02"] * 3), "diverge"),
])
def test_validacao_rejeita_violacoes(alteracao, mensagem):
    with pytest.raises(ErroValidacaoDados, match=mensagem):
        validar_bruto(alteracao(_bruto_valido()), "ruim.csv", CFG)


def test_agregar_diario_preenche_zeros_e_marca_sem_registro():
    t = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-03"]), "money": [10.0, 20.0, 5.0]})
    d = agregar_diario(t, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-04"))
    assert d["demanda"].tolist() == [2, 0, 1, 0]
    assert d["receita"].tolist() == [30.0, 0.0, 5.0, 0.0]
    assert d["sem_registro"].tolist() == [False, True, False, True]
    assert d.index.freq == "D"


def test_contagem_de_registros(base):
    assert (base.transacoes["maquina"] == 1).sum() == 3636
    assert (base.transacoes["maquina"] == 2).sum() == 262


def test_inicio_das_series(base):
    assert base.inicio["Máquina 1"] == pd.Timestamp("2024-03-01")
    assert base.inicio["Máquina 2"] == pd.Timestamp("2025-02-08")
    assert base.fim == pd.Timestamp("2025-03-23")


def test_calendario_continuo_sem_ausentes(base):
    for (serie, alvo), s in base.series.items():
        assert s.index.freq == "D", (serie, alvo)
        assert s.notna().all() and (s >= 0).all()
        assert (s.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_total_igual_soma_das_maquinas(base):
    for alvo in ("demanda", "receita"):
        soma = base.series[("Máquina 1", alvo)] + base.series[("Máquina 2", alvo)].reindex(base.series[("Total", alvo)].index).fillna(0)
        assert np.allclose(soma, base.series[("Total", alvo)])


def test_zero_de_demanda_implica_zero_de_receita(base):
    for serie in ("Máquina 1", "Máquina 2", "Total"):
        d, r = base.series[(serie, "demanda")], base.series[(serie, "receita")]
        assert ((d == 0) == (r == 0)).all()


def test_maquina_2_nao_tem_zeros_antes_de_existir(base):
    """Antes de 08/02/2025 a M2 não existe: a série não começa com zeros fictícios."""
    assert base.series[("Máquina 2", "demanda")].index.min() == base.inicio["Máquina 2"]
    assert base.series[("Máquina 2", "demanda")].iloc[0] > 0


def test_card_nulo_somente_em_dinheiro_na_maquina_1(base):
    m1 = base.transacoes[base.transacoes["maquina"] == 1]
    assert (m1.loc[m1["card"].isna(), "cash_type"] == "cash").all()
    assert m1.loc[m1["cash_type"] == "cash", "card"].isna().all()


def test_maquina_2_sem_identificador_de_cartao(base):
    assert base.transacoes.loc[base.transacoes["maquina"] == 2, "card"].isna().all()


def test_manifesto_de_hash_dos_dados(base):
    import json

    from src.settings import caminho

    manifesto = json.loads((caminho("dados_brutos") / "MANIFEST.json").read_text())
    assert set(manifesto["arquivos"]) == {"index_1.csv", "index_2.csv"}
    assert manifesto["arquivos"]["index_1.csv"]["sha256"].startswith("2a600445")


def test_dados_brutos_conferem_com_o_manifesto():
    """Os bytes em data/raw são exatamente os do manifesto versionado (pré-condição da reprodutibilidade)."""
    import json

    from src.data.load_data import sha256
    from src.settings import caminho

    pasta = caminho("dados_brutos")
    manifesto = json.loads((pasta / "MANIFEST.json").read_text())["arquivos"]
    for arquivo, info in manifesto.items():
        assert sha256(pasta / arquivo) == info["sha256"], arquivo


def test_manifesto_divergente_interrompe_a_carga(tmp_path):
    """Um arquivo alterado em data/raw não pode passar em silêncio."""
    import json
    import shutil

    from src.data.load_data import garantir_dados_brutos
    from src.settings import carregar_config, caminho

    config = carregar_config()
    origem = caminho("dados_brutos", config)
    for arquivo in ("index_1.csv", "index_2.csv", "MANIFEST.json"):
        shutil.copy2(origem / arquivo, tmp_path / arquivo)
    with open(tmp_path / "index_2.csv", "a", encoding="utf-8") as f:
        f.write("2025-03-23,2025-03-23 23:59:59.000,cash,,30.0,Latte\n")
    config = json.loads(json.dumps(config))
    config["caminhos"]["dados_brutos"] = str(tmp_path)
    with pytest.raises(ErroValidacaoDados, match="MANIFEST"):
        garantir_dados_brutos(config)
