"""Padronização das transações e construção das séries diárias.

Semântica adotada (reaproveitada de ``modelagens_testes.ipynb``, seção 2.1):

| Situação | Interpretação | Tratamento |
|---|---|---|
| Dia **dentro** do período ativo da máquina sem transação | zero vendas registradas | demanda = 0 e receita = 0 (não é imputado) |
| Dia **antes** da primeira venda da Máquina 2 (< 08/02/2025) | máquina inexistente/não registrada | fora da série da Máquina 2 (não é zero nem missing) |
| Total antes de 08/02/2025 | só a Máquina 1 existe | Total = Máquina 1; a entrada vira ``maquinas_ativas`` |

Nenhum registro é removido além dos inválidos (não há nenhum na versão 21). Registros com o
mesmo timestamp são mantidos: na Máquina 2 eles correspondem a pedidos com vários itens.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.logging_config import obter_logger
from src.settings import carregar_config, caminho

log = obter_logger(__name__)


@dataclass
class BaseDados:
    """Tudo o que as etapas seguintes precisam dos dados.

    Attributes:
        transacoes: uma linha por venda, padronizada.
        diarios: por série, DataFrame diário com ``demanda``, ``receita`` e ``sem_registro``.
        series: ``{(série, alvo): pd.Series}`` com frequência diária contínua.
        inicio: primeira data de cada série.
        fim: última data observada (comum às três séries).
    """
    transacoes: pd.DataFrame
    diarios: dict[str, pd.DataFrame]
    series: dict[tuple[str, str], pd.Series]
    inicio: dict[str, pd.Timestamp]
    fim: pd.Timestamp
    resumo_carga: pd.DataFrame = field(default_factory=pd.DataFrame)


def padronizar_transacoes(bruto: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Padroniza textos, tipos e cria colunas derivadas (data, hora, produto padronizado).

    O nome do produto é normalizado (minúsculas + primeira maiúscula) porque os dois arquivos
    grafam o mesmo item de formas diferentes (``Americano with Milk`` × ``Americano with milk``).
    """
    config = config or carregar_config()
    df = bruto.copy()
    for coluna in config["dados"]["colunas_texto"]:
        if coluna in df.columns:
            df[coluna] = df[coluna].astype("string").str.strip()
    df["money"] = pd.to_numeric(df["money"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", format="mixed")
    antes = len(df)
    df = df.dropna(subset=["datetime", "money"]).copy()
    if len(df) < antes:
        log.warning("%d registros removidos por datetime/money inválidos", antes - len(df))
    df["date"] = df["datetime"].dt.normalize()
    df["hora"] = df["datetime"].dt.hour
    df["dia_semana"] = df["datetime"].dt.dayofweek
    df["produto"] = df["coffee_name"].str.lower().str.capitalize()
    df["serie"] = "Máquina " + df["maquina"].astype(str)
    return df.sort_values("datetime").reset_index(drop=True)


def agregar_diario(dados: pd.DataFrame, inicio: pd.Timestamp, fim: pd.Timestamp) -> pd.DataFrame:
    """Agrega transações em calendário diário contínuo de ``inicio`` a ``fim``.

    Dias sem registro recebem demanda e receita zero e ``sem_registro = True``.
    """
    calendario = pd.date_range(inicio, fim, freq="D")
    diario = (dados.groupby("date").agg(demanda=("money", "size"), receita=("money", "sum"))
              .reindex(calendario))
    diario["sem_registro"] = diario["demanda"].isna()
    diario[["demanda", "receita"]] = diario[["demanda", "receita"]].fillna(0.0)
    diario.index.name = "data"
    return diario.asfreq("D")


def construir_base(transacoes: pd.DataFrame, config: dict | None = None) -> BaseDados:
    """Constrói as séries diárias das máquinas e do Total consolidado."""
    config = config or carregar_config()
    fim = transacoes["date"].max()
    inicio = {f"Máquina {k}": transacoes.loc[transacoes["maquina"] == k, "date"].min()
              for k in sorted(transacoes["maquina"].unique())}
    diarios = {m: agregar_diario(transacoes[transacoes["serie"] == m], inicio[m], fim) for m in inicio}
    m1 = diarios["Máquina 1"]
    m2 = diarios["Máquina 2"].reindex(m1.index)
    total = pd.DataFrame({"demanda": m1["demanda"] + m2["demanda"].fillna(0),
                          "receita": m1["receita"] + m2["receita"].fillna(0)}).asfreq("D")
    total["sem_registro"] = total["demanda"] == 0
    total["maquinas_ativas"] = 1 + m2["demanda"].notna().astype(int)
    diarios["Total"] = total
    inicio["Total"] = m1.index.min()
    series = {(m, a): diarios[m][a].astype(float).asfreq("D").rename(a)
              for m in config["series"]["nomes"] for a in config["series"]["alvos"]}

    resumo = pd.DataFrame({
        "arquivo": transacoes.groupby("serie")["arquivo_origem"].first(),
        "registros": transacoes.groupby("serie").size(),
        "primeira data": transacoes.groupby("serie")["date"].min().dt.strftime("%d/%m/%Y"),
        "última data": transacoes.groupby("serie")["date"].max().dt.strftime("%d/%m/%Y"),
        "dias no período ativo": {m: len(d) for m, d in diarios.items() if m != "Total"},
        "dias sem venda": {m: int(d["sem_registro"].sum()) for m, d in diarios.items() if m != "Total"},
        "duplicatas exatas": transacoes.groupby("serie").apply(
            lambda g: int(g.duplicated(subset=["datetime", "money", "coffee_name", "cash_type"]).sum()),
            include_groups=False),
    })
    base = BaseDados(transacoes, diarios, series, inicio, fim, resumo)
    validar_base(base)
    log.info("Séries diárias construídas: %s", {k: len(v) for k, v in series.items()})
    return base


def validar_base(base: BaseDados) -> None:
    """Invariantes das séries diárias (falham cedo se o pré-processamento mudar).

    - calendário contínuo e sem valores ausentes;
    - demanda e receita não negativas; receita zero ⇔ demanda zero;
    - Total = Máquina 1 + Máquina 2 em todos os dias.
    """
    for (m, a), s in base.series.items():
        assert s.index.freq == "D" and s.notna().all(), f"{m}/{a}: calendário com lacunas"
        assert (s >= 0).all(), f"{m}/{a}: valores negativos"
    for m, d in base.diarios.items():
        assert ((d["demanda"] == 0) == (d["receita"] == 0)).all(), f"{m}: receita e demanda incoerentes"
    for a in ("demanda", "receita"):
        soma = base.series[("Máquina 1", a)] + base.series[("Máquina 2", a)].reindex(base.series[("Total", a)].index).fillna(0)
        assert np.allclose(soma, base.series[("Total", a)]), f"Total ≠ M1 + M2 em {a}"


def salvar_processados(base: BaseDados, config: dict | None = None) -> None:
    """Grava as transações padronizadas e as séries diárias em ``data/processed``."""
    pasta = caminho("dados_processados", config)
    base.transacoes.to_csv(pasta / "transacoes_padronizadas.csv", index=False)
    for nome, d in base.diarios.items():
        d.to_csv(pasta / f"diario_{carregar_config()['series']['pastas'][nome]}.csv")
    base.resumo_carga.to_csv(pasta / "resumo_carga.csv")
    log.info("Dados processados gravados em %s", pasta)


def preparar_base(config: dict | None = None, salvar: bool = False) -> BaseDados:
    """Atalho: carga + validação + padronização + séries diárias."""
    from src.data.load_data import carregar_brutos

    config = config or carregar_config()
    base = construir_base(padronizar_transacoes(carregar_brutos(config), config), config)
    if salvar:
        salvar_processados(base, config)
    return base
