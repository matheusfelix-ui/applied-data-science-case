"""Modelo de Machine Learning candidato: Random Forest com previsão recursiva.

Portado de ``modelagens_testes.ipynb`` (seções 5.2 e 6.5):

- **Atributos** do ``ConstrutorAtributos`` (defasagens, janelas móveis, calendário, tendência,
  dummies de quebra conhecidas na origem e indicadores comportamentais defasados).
- **Seleção em dois estágios** nas dobras internas, com a mesma previsão recursiva de 14 dias
  usada no teste (critério: WAPE agregado das dobras): (1) profundidade, folha mínima e fração de
  atributos; (2) meia-vida dos pesos de recência {sem ponderação, 120, 60, 30 dias} × inclusão dos
  atributos comportamentais (o grupo de atributos também é escolhido pela validação, não pelo teste).
  Em cada estágio vale a regra de 1 erro-padrão: a configuração menos complexa dentro de 1 EP do melhor.
- **Previsão recursiva**: no horizonte, ``lag_1`` de T+2 é a previsão de T+1 etc.; nunca usa o teste.
- **Intervalos conformais**: quantis de |erro| das previsões de validação, só das dobras posteriores
  à última quebra conhecida (se houver ≥ 10 erros), para que sejam permutáveis com o teste.
- **Ajuste honesto**: valores ajustados = previsões de 1 passo fora da amostra, com reestimação a
  cada 28 dias (o ajuste dentro da amostra de uma floresta é memorização e não informa nada).

Por que Random Forest e não boosting: nas 11 origens de ``modelagens_testes`` (seção 8.5), LightGBM
e XGBoost ficaram na metade de baixo em 3 de 4 séries; a floresta ficou entre os 5 primeiros nas
4 séries, é robusta a hiperparâmetros e não exige biblioteca nativa extra (libomp).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.features.feature_pipeline import ConstrutorAtributos
from src.features.lag_features import cauda_necessaria
from src.models.model_selection import UM_DIA, Contexto, NaoAplicavel, intervalos_conformais
from src.settings import carregar_config

_CFG = carregar_config()
_RF = _CFG["modelos"]["random_forest"]
SEMENTE = _CFG["projeto"]["semente"]
N_JOBS = {"valor": _CFG["modelos"]["n_jobs"]}
REGRA = {"valor": _RF["regra_selecao"]}


def definir_regra(regra: str) -> None:
    """Regra de escolha na validação: "1ep" (1 erro-padrão) ou "minimo" (menor WAPE)."""
    assert regra in ("1ep", "minimo")
    REGRA["valor"] = regra


def definir_n_jobs(n: int) -> None:
    """Threads por floresta (1 nos processos paralelos do backtest)."""
    N_JOBS["valor"] = n


def grade_estagio_1() -> list[dict]:
    g = _RF["grade"]
    return [dict(max_depth=d, min_samples_leaf=l, max_features=f)
            for d, l, f in itertools.product(g["max_depth"], g["min_samples_leaf"], g["max_features"])]


def _fabricar(params: dict) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=_RF["n_estimators"], bootstrap=True, random_state=SEMENTE,
                                 n_jobs=N_JOBS["valor"], **params)


def _pesos(datas: pd.DatetimeIndex, origem: pd.Timestamp, meia_vida) -> np.ndarray:
    """Peso 1 na origem, metade a cada ``meia_vida`` dias para trás (None = pesos iguais)."""
    if meia_vida is None:
        return np.ones(len(datas))
    return 0.5 ** ((origem - datas).days.to_numpy() / meia_vida)


PARAMS_NAO_ARVORE = ("meia_vida", "comportamentais")


def _params_arvore(config: dict) -> dict:
    return {k: v for k, v in config.items() if k not in PARAMS_NAO_ARVORE}


def _dados_treino(ctx: Contexto, origem: pd.Timestamp, comportamentais: bool = True):
    y = ctx.base.series[(ctx.serie, ctx.alvo)].loc[:origem]
    construtor = ConstrutorAtributos(ctx.base, ctx.detector, ctx.serie, origem, comportamentais)
    return y, construtor, construtor(y)


def prever_recursivo(modelo, colunas: list[str], historico: pd.Series, construtor, datas_futuras) -> pd.Series:
    """Previsão recursiva: cada passo usa o observado até a origem + previsões anteriores."""
    serie = historico.copy()
    cauda = cauda_necessaria()
    saida = []
    for d in datas_futuras:
        serie.loc[d] = np.nan
        X = construtor(serie.iloc[-cauda:]).loc[[d], colunas]
        p = max(float(modelo.predict(X)[0]), 0.0)
        serie.loc[d] = p
        saida.append(p)
    return pd.Series(saida, index=datas_futuras)


def treinar_prever(ctx: Contexto, config: dict, origem: pd.Timestamp, datas_futuras):
    """Treina com dados até ``origem`` e prevê ``datas_futuras`` recursivamente."""
    params = _params_arvore(config)
    y, construtor, X = _dados_treino(ctx, origem, config.get("comportamentais", True))
    if len(y) < _CFG["divisao"]["min_dias_treino_validacao"]:
        raise NaoAplicavel("histórico insuficiente")
    modelo = _fabricar(params)
    modelo.fit(X, y.to_numpy(), sample_weight=_pesos(y.index, origem, config.get("meia_vida")))
    return modelo, list(X.columns), prever_recursivo(modelo, list(X.columns), y, construtor, datas_futuras)


def _avaliar(ctx: Contexto, config: dict) -> tuple[float, dict]:
    erro, soma, previsoes = 0.0, 0.0, {}
    for o in ctx.origens_validacao():
        datas = pd.date_range(o + UM_DIA, periods=ctx.horizonte)
        _, _, p = treinar_prever(ctx, config, o, datas)
        obs = ctx.y_treino.loc[datas]
        erro += np.abs(obs - p).sum()
        soma += obs.abs().sum()
        previsoes[o] = p
    return (100 * erro / soma if soma else np.nan), previsoes


def _rotulo(config: dict) -> str:
    partes = []
    for k, v in config.items():
        if k == "meia_vida":
            partes.append("sem ponderação" if v is None else f"meia-vida={v} d")
        elif k == "comportamentais":
            partes.append("com atributos comportamentais" if v else "sem atributos comportamentais")
        else:
            partes.append(f"{k}={v}")
    return ", ".join(partes)


def complexidade(config: dict) -> int:
    """Escore de complexidade (maior = mais flexível/mais variância) para a regra de 1 erro-padrão."""
    return int(sum([config.get("max_depth") is None, config.get("min_samples_leaf") == 1,
                    config.get("max_features") == 1.0, config.get("meia_vida") is not None,
                    bool(config.get("comportamentais", False))]))


def _escolher_1ep(candidatos: list[tuple]) -> tuple:
    """Regra de 1 erro-padrão (Breiman et al., 1984; ESL §7.10).

    Entre as configurações cujo WAPE de validação está a até 1 EP do melhor, escolhe a menos
    complexa (desempate pelo WAPE). Com poucas dobras (a Máquina 2 tem uma só), diferenças menores
    que 1 EP são ruído, e preferir o modelo mais regularizado reduz o sobreajuste da seleção.
    """
    melhor = min(candidatos, key=lambda c: c[0])
    if REGRA["valor"] == "minimo":
        return melhor
    limite = melhor[0] + melhor[3]
    elegiveis = [c for c in candidatos if c[0] <= limite]
    return min(elegiveis, key=lambda c: (complexidade(c[1]), c[0]))


def selecionar(ctx: Contexto) -> tuple[dict, pd.DataFrame, dict]:
    """Seleção em dois estágios (WAPE agregado das dobras internas) com a regra de 1 erro-padrão."""
    avaliados, linhas = {}, []

    def avaliar(config, estagio):
        chave = _rotulo(config)
        if chave not in avaliados:
            wape, prev = _avaliar(ctx, config)
            erros = np.concatenate([np.abs(ctx.y_treino.loc[p.index] - p) for p in prev.values()])
            soma_y = sum(ctx.y_treino.loc[p.index].sum() for p in prev.values())
            ep = 100 * erros.std(ddof=1) * np.sqrt(len(erros)) / soma_y if soma_y else np.nan
            avaliados[chave] = (wape, config, prev, ep, estagio)
            linhas.append({"estágio": estagio, "configuração": chave, "WAPE validação %": wape, "EP do WAPE": ep,
                           "complexidade": complexidade(config)})
        return avaliados[chave]

    estagio_1 = [avaliar({**p, "meia_vida": None, "comportamentais": True}, 1) for p in grade_estagio_1()]
    melhor_1 = _params_arvore(_escolher_1ep(estagio_1)[1])
    estagio_2 = [avaliar({**melhor_1, "meia_vida": mv, "comportamentais": comp}, 2)
                 for mv, comp in itertools.product(_RF["meias_vidas"], (True, False))]
    _, config, previsoes, _, _ = _escolher_1ep(estagio_2)
    grade = pd.DataFrame(linhas).sort_values("WAPE validação %").reset_index(drop=True)
    grade["escolhida"] = grade["configuração"] == _rotulo(config)
    return config, grade, previsoes


def ajuste_fora_da_dobra(ctx: Contexto, config: dict, bloco: int | None = None) -> pd.Series:
    """Previsões de 1 passo fora da amostra, reestimando o modelo a cada ``bloco`` dias."""
    bloco = bloco or _CFG["divisao"]["bloco_ajuste_fora_dobra"]
    params = _params_arvore(config)
    y_total = ctx.y_treino
    ajustado = pd.Series(np.nan, index=y_total.index)
    inicio = y_total.index.min() + pd.Timedelta(days=_CFG["divisao"]["dias_aquecimento"][ctx.serie])
    for b in pd.date_range(inicio, ctx.origem, freq=f"{bloco}D"):
        fim_bloco = min(b + pd.Timedelta(days=bloco - 1), ctx.origem)
        y, construtor, X = _dados_treino(ctx, b - UM_DIA, config.get("comportamentais", True))
        if len(y) < _CFG["divisao"]["min_dias_treino_validacao"]:
            continue
        m = _fabricar(params)
        m.fit(X, y.to_numpy(), sample_weight=_pesos(y.index, b - UM_DIA, config.get("meia_vida")))
        Xb = construtor(y_total.loc[:fim_bloco]).loc[b:fim_bloco, list(X.columns)]
        ajustado.loc[Xb.index] = np.clip(m.predict(Xb), 0, None)
    return ajustado


def modelo_random_forest(ctx: Contexto) -> dict:
    """Random Forest individual por série, com seleção interna, recursão e intervalos conformais."""
    if len(ctx.origens_validacao()) == 0:
        raise NaoAplicavel("sem dobras de validação")
    config, grade, prev_val = selecionar(ctx)
    modelo, colunas, previsao = treinar_prever(ctx, config, ctx.origem, ctx.datas_previsao)
    ajustado = ajuste_fora_da_dobra(ctx, config) if ctx.cenario in ("principal", "final") else pd.Series(dtype=float)
    ultima = max(ctx.quebras) if ctx.quebras else None
    residuos = [ctx.y_treino.loc[p.index] - p for o, p in prev_val.items() if ultima is None or o >= ultima]
    residuos = pd.concat(residuos) if residuos else pd.Series(dtype=float)
    regra = f"dobras após {ultima:%d/%m/%y}" if ultima is not None else "todas as dobras"
    if len(residuos) < _CFG["modelos"]["conformal_min_residuos"]:
        residuos = pd.concat([ctx.y_treino.loc[p.index] - p for p in prev_val.values()])
        regra = "todas as dobras"
    mae_val = float(np.mean(np.concatenate([np.abs(ctx.y_treino.loc[p.index] - p) for p in prev_val.values()])))
    importancias = pd.Series(modelo.feature_importances_, index=colunas).sort_values(ascending=False)
    return {"previsao": previsao.to_numpy(), "ajustado": ajustado, "intervalos": intervalos_conformais(previsao, residuos),
            "especificacao": _rotulo(config), "mae_validacao": mae_val, "modelo": modelo,
            "extras": {"grade": grade, "config": config, "importancias": importancias, "colunas": colunas,
                       "erros do intervalo": regra, "n_residuos": len(residuos), "previsoes_validacao": prev_val}}
