"""Dummies das quebras estruturais, construídas **só com a informação disponível na origem**.

| Variável | Definição | Fundamentação (``03_structural_breaks``) |
|---|---|---|
| ``dummy_horario_reduzido`` | 1 do início da série até a véspera da extensão da janela de vendas; 0 depois | Binseg + BIC na proporção de vendas fora de 10h–19h (18/05/2024 só com treino; 14/05 com a série inteira); sup-Wald global da demanda da M1 estima a maior quebra de nível exatamente em 14/05/2024 |
| ``dummy_novos_clientes`` | 1 a partir da quebra de patamar de 2025 | Binseg + BIC na demanda (30/01/2025 na M1, 04/02/2025 no Total), confirmada por sup-Wald local; 177 dos 196 cartões ativos depois dela são novos |
| ``regime_<data>`` | 1 a partir de cada quebra de nível intermediária (24/07/2024, 16/11/2024) | candidatas: cada modelo decide se as usa |
| ``maquinas_ativas`` (Total) | 1 antes de 08/02/2025, 2 depois | fato operacional: entrada da Máquina 2 |

Em cada origem do backtest as datas são **redetectadas com os dados até ela**: antes de uma
quebra ser detectável, a dummy correspondente simplesmente não existe (é 0), sem vazamento.
Para datas futuras, supõe-se que o regime vigente na origem persiste no horizonte.
"""
from __future__ import annotations

import pandas as pd

from src.structural_breaks.structural_break_test import DetectorQuebras


def dummies_quebras(datas: pd.DatetimeIndex, serie: str, origem: pd.Timestamp, detector: DetectorQuebras,
                    inicio_m2: pd.Timestamp) -> pd.DataFrame:
    """Dummies de quebra conhecidas em ``origem`` para as ``datas`` pedidas (passadas ou futuras)."""
    X = pd.DataFrame(index=datas)
    janela = detector.janela_ate(serie, origem)
    X["dummy_horario_reduzido"] = (datas < janela).astype(int) if janela is not None else 0
    novos = detector.quebra_novos_clientes_ate(serie, origem)
    X["dummy_novos_clientes"] = (datas >= novos).astype(int) if novos is not None else 0
    for q in detector.quebras_ate(serie, origem):
        if novos is not None and q == novos:
            continue
        X[f"regime_{q:%d%m%y}"] = (datas >= q).astype(int)
    if serie == "Total":
        X["maquinas_ativas"] = 1 + (datas >= inicio_m2).astype(int)
    return X


def colunas_quebra(X: pd.DataFrame) -> list[str]:
    """Colunas de quebra/regime presentes num DataFrame de atributos."""
    return [c for c in X.columns if c.startswith(("dummy_", "regime_")) or c == "maquinas_ativas"]
