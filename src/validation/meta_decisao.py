"""Meta-decisão da regra de seleção da Random Forest, tomada **antes** de olhar o teste.

A Random Forest escolhe a configuração na validação interna por uma de duas regras:
- ``"1ep"``: a configuração mais simples cujo erro fica a até 1 erro-padrão do melhor;
- ``"minimo"``: a configuração de menor erro.

Para não escolher a regra olhando o teste, as duas foram comparadas no backtest das 10 origens
**anteriores** ao teste (Máquina 1 e Total, demanda e receita). Houve empate prático (MASE médio
0,862 × 0,868), e a regra mais simples (``"1ep"``) ficou registrada em ``config.yaml``
(``modelos.random_forest.regra_selecao``).

Uso (fora do pipeline principal, porque só documenta uma decisão já tomada; ~5 min)::

    python -m src.validation.meta_decisao

Grava ``reports/model_comparison/meta_decisao_regra_rf_backtest_pre_teste.csv``.
"""
from __future__ import annotations

import pandas as pd

from src.data.preprocessing import BaseDados, preparar_base
from src.logging_config import obter_logger
from src.settings import pasta_relatorio
from src.validation.backtesting import backtest, datas_divisao, tabela_resultados

log = obter_logger(__name__)

REGRAS = ("1ep", "minimo")
SERIES = [(s, a) for s in ("Máquina 1", "Total") for a in ("demanda", "receita")]   # a Máquina 2 não tem histórico para backtest
MODELOS = ["Random Forest", "Média 4 semanas", "Naive sazonal"]


def meta_decisao_regra_rf(base: BaseDados) -> pd.DataFrame:
    """MAE, MASE e viés (ME) médios da Random Forest nas 10 origens anteriores ao teste, por regra de seleção.

    Args:
        base: dados preparados (``preparar_base``).

    Returns:
        Tabela com índice (série, alvo) e colunas (métrica, regra).
    """
    origens = datas_divisao(base)["origens_backtest"]
    tabelas = {}
    for regra in REGRAS:
        resultados, _ = backtest(base, origens, SERIES, MODELOS, regra_rf=regra)
        tabelas[regra] = tabela_resultados(resultados, base)
    t = pd.concat(tabelas, names=["regra"]).reset_index(level=0)
    t = t[t["modelo"] == "Random Forest"]
    return t.groupby(["série", "alvo", "regra"])[["MAE", "MASE", "ME"]].mean().unstack("regra").round(3)


def main() -> None:
    tabela = meta_decisao_regra_rf(preparar_base())
    destino = pasta_relatorio("model_comparison") / "meta_decisao_regra_rf_backtest_pre_teste.csv"
    tabela.to_csv(destino)
    log.info("Meta-decisão da regra da RF gravada em %s", destino)
    print(tabela)


if __name__ == "__main__":
    main()
