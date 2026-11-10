"""Infraestrutura comum dos modelos: contexto da origem, seleção interna e intervalos.

Reaproveita a estrutura de ``modelagens_testes.ipynb`` (seção 5):

- ``Contexto`` contém **apenas** o que um modelo pode ver na origem: a série até T, as datas a
  prever, as quebras e a janela de vendas detectadas até T. O teste nunca está no contexto.
- Seleção interna por validação temporal: até 3 dobras de 14 dias dentro do treino (janela
  expansiva); uma dobra só é usada se houver pelo menos 14 dias de treino até a sua origem.
- Janela de estimação como hiperparâmetro (treino inteiro, desde a última quebra, últimas 16 ou
  8 semanas), sempre relativa à origem de cada dobra.
- Transformação de Anscombe z = √(y + 3/8) para contagens, com volta à escala original corrigida.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import stats

from src.data.preprocessing import BaseDados
from src.features.break_dummies import dummies_quebras
from src.features.temporal_features import features_calendario
from src.settings import carregar_config
from src.structural_breaks.structural_break_test import DetectorQuebras

_CFG = carregar_config()
UM_DIA = pd.Timedelta(days=1)
JANELAS = _CFG["modelos"]["janelas_estimacao"]
TRANSFORMACOES = _CFG["modelos"]["transformacoes"]


class NaoAplicavel(Exception):
    """O modelo não se aplica a esta série/origem (o motivo é registrado)."""


@dataclass
class Contexto:
    """Informação disponível na origem ``origem`` para prever ``serie``/``alvo``."""
    base: BaseDados
    detector: DetectorQuebras
    serie: str
    alvo: str
    origem: pd.Timestamp
    cenario: str = "principal"
    horizonte: int = _CFG["divisao"]["horizonte"]

    @cached_property
    def y_treino(self) -> pd.Series:
        return self.base.series[(self.serie, self.alvo)].loc[:self.origem].copy()

    @property
    def fim_treino(self) -> pd.Timestamp:
        return self.y_treino.index.max()

    @property
    def datas_previsao(self) -> pd.DatetimeIndex:
        return pd.date_range(self.origem + UM_DIA, periods=self.horizonte, freq="D")

    @property
    def quebras(self) -> list[pd.Timestamp]:
        return self.detector.quebras_ate(self.serie, self.origem)

    def exogenas(self, datas: pd.DatetimeIndex) -> pd.DataFrame:
        """Exógenas determinísticas conhecidas na origem: calendário e dummies de quebra."""
        cal = features_calendario(datas)[["feriado", "data_comercial", "inicio_mes"]]
        return cal.join(dummies_quebras(datas, self.serie, self.origem, self.detector, self.base.inicio["Máquina 2"]))

    def origens_validacao(self) -> list[pd.Timestamp]:
        """Origens das dobras internas (blocos de 14 dias dentro do treino, janela expansiva)."""
        d = _CFG["divisao"]
        origens = [self.fim_treino - pd.Timedelta(days=self.horizonte * k) for k in range(d["n_dobras_validacao"], 0, -1)]
        return [o for o in origens if (o - self.base.inicio[self.serie]).days + 1 >= d["min_dias_treino_validacao"]]

    def sub(self, origem: pd.Timestamp, alvo: str | None = None) -> "Contexto":
        """Contexto de uma dobra (ou de outro alvo) com treino até ``origem``."""
        return Contexto(self.base, self.detector, self.serie, alvo or self.alvo, origem, f"{self.cenario} | validação",
                        self.horizonte)


@dataclass
class ResultadoModelo:
    """Saída padronizada de um modelo numa série/origem."""
    modelo: str
    familia: str
    serie: str
    alvo: str
    cenario: str
    origem: pd.Timestamp
    previsao: pd.Series
    ajustado: pd.Series
    intervalos: pd.DataFrame | None
    especificacao: str
    inicio_treino: pd.Timestamp
    tempo_s: float
    mae_validacao: float = np.nan
    extras: dict = field(default_factory=dict)


def erro_validacao(ctx: Contexto, prever) -> float:
    """MAE médio nas dobras internas; ``prever(sub_ctx)`` devolve as previsões do horizonte."""
    erros = []
    for origem in ctx.origens_validacao():
        sub = ctx.sub(origem)
        try:
            previsto = np.clip(np.asarray(prever(sub), dtype=float), 0, None)
        except NaoAplicavel:
            raise
        except Exception:
            return np.nan
        erros.append(np.abs(ctx.y_treino.loc[sub.datas_previsao].to_numpy() - previsto))
    return float(np.mean(np.concatenate(erros))) if erros else np.nan


def inicio_janela(ctx: Contexto, janela: str) -> pd.Timestamp:
    """Início da janela de estimação, relativo à origem do contexto."""
    inicio = ctx.y_treino.index.min()
    if janela == "todo o treino":
        return inicio
    if janela == "desde a última quebra":
        return max(ctx.quebras) if ctx.quebras else inicio
    semanas = {"últimas 16 semanas": 16, "últimas 8 semanas": 8}[janela]
    return max(inicio, ctx.fim_treino - pd.Timedelta(days=7 * semanas - 1))


def janelas_candidatas(ctx: Contexto) -> list[str]:
    """Janelas com inícios distintos (na Máquina 2 todas coincidem com o treino inteiro)."""
    vistas, saida = set(), []
    for j in JANELAS:
        ini = inicio_janela(ctx, j)
        if ini not in vistas:
            vistas.add(ini)
            saida.append(j)
    return saida


def transformar(y, transformacao: str) -> np.ndarray:
    """Anscombe z = √(y + 3/8) ou identidade."""
    y = np.asarray(y, float)
    return np.sqrt(y + 3 / 8) if transformacao == "raiz (Anscombe)" else y


def media_original(media_z, variancia_z, transformacao: str) -> np.ndarray:
    """Volta da média à escala original com correção de viés: E[y] = E[z]² + Var(z) − 3/8."""
    media_z = np.asarray(media_z, float)
    if transformacao != "raiz (Anscombe)":
        return media_z
    return np.clip(media_z, 0, None) ** 2 + np.asarray(variancia_z, float) - 3 / 8


def quantil_original(q_z, transformacao: str) -> np.ndarray:
    """Quantis são preservados por transformações monótonas: basta inverter."""
    q_z = np.asarray(q_z, float)
    return np.clip(q_z, 0, None) ** 2 - 3 / 8 if transformacao == "raiz (Anscombe)" else q_z


def intervalos_normais(media, desvio) -> pd.DataFrame:
    """Intervalos gaussianos de 80% e 95%."""
    media, desvio = np.asarray(media, float), np.asarray(desvio, float)
    saida = {}
    for nivel in (80, 95):
        z = stats.norm.ppf(0.5 + nivel / 200)
        saida[f"li_{nivel}"], saida[f"ls_{nivel}"] = media - z * desvio, media + z * desvio
    return pd.DataFrame(saida)


def intervalos_conformais(previsao, residuos, minimo: int | None = None) -> pd.DataFrame | None:
    """Intervalo simétrico: previsão ± quantil (com correção de amostra finita) de |erro| de validação."""
    minimo = minimo or _CFG["modelos"]["conformal_min_residuos"]
    r = np.abs(np.asarray(residuos, float))
    if len(r) < minimo:
        return None
    previsao = np.asarray(previsao, float)
    saida = {}
    for nivel in (80, 95):
        q = np.quantile(r, min(1.0, np.ceil((len(r) + 1) * nivel / 100) / len(r)))
        saida[f"li_{nivel}"], saida[f"ls_{nivel}"] = np.clip(previsao - q, 0, None), previsao + q
    return pd.DataFrame(saida)
