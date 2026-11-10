"""Métricas de previsão e piso de ruído de contagem.

Convenções:
- ``ME = média(previsto − observado)``: ME > 0 indica **superestimação** (viés).
- MAPE só é calculado se não houver zeros no período avaliado (seria indefinido).
- WAPE = Σ|erro| / Σ|observado|: robusto a zeros, interpretável como "% do volume".
- MASE: MAE ÷ MAE do naive sazonal (m = 7) dentro do treino; < 1 bate o naive in-sample.
- Intervalos: cobertura de 80% e 95% e *Winkler score* (largura + penalidade por fora).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calcular_metricas(y, previsto, y_treino=None, intervalos: pd.DataFrame | None = None) -> dict:
    """MAE, RMSE, MAPE, sMAPE, WAPE, R², ME, ME relativo, MASE e métricas de intervalo.

    R² = 1 − SQE / SQT em torno da média do próprio período avaliado: mede quanto da variação dia a
    dia a previsão acompanha. Em 14 dias de contagens ruidosas ele é instável e pode ser negativo
    (previsão pior que a média do período, que não é conhecida de antemão).
    """
    y, previsto = np.asarray(y, float), np.asarray(previsto, float)
    erro = previsto - y
    den = np.abs(y) + np.abs(previsto)
    r = {
        "MAE": float(np.mean(np.abs(erro))),
        "RMSE": float(np.sqrt(np.mean(erro ** 2))),
        "MAPE_%": float(np.mean(np.abs(erro) / y) * 100) if (y > 0).all() else np.nan,
        "sMAPE_%": float(np.mean(np.where(den > 0, 2 * np.abs(erro) / np.where(den > 0, den, 1), 0)) * 100),
        "WAPE_%": float(np.sum(np.abs(erro)) / np.sum(np.abs(y)) * 100) if np.sum(np.abs(y)) > 0 else np.nan,
        "R2": float(1 - np.sum(erro ** 2) / np.sum((y - y.mean()) ** 2)) if np.sum((y - y.mean()) ** 2) > 0 else np.nan,
        "ME": float(np.mean(erro)),
        "ME_relativo_%": float(np.mean(erro) / np.mean(y) * 100) if np.mean(y) > 0 else np.nan,
    }
    if y_treino is not None:
        v = np.asarray(y_treino, float)
        escala = np.mean(np.abs(v[7:] - v[:-7])) if len(v) > 7 else np.nan
        r["MASE"] = float(r["MAE"] / escala) if escala and np.isfinite(escala) else np.nan
    if intervalos is not None:
        for nivel in (80, 95):
            li, ls = intervalos[f"li_{nivel}"].to_numpy(float), intervalos[f"ls_{nivel}"].to_numpy(float)
            r[f"cobertura_{nivel}_%"] = float(np.mean((y >= li) & (y <= ls)) * 100)
        li, ls = intervalos["li_95"].to_numpy(float), intervalos["ls_95"].to_numpy(float)
        r["largura_95"] = float(np.mean(ls - li))
        r["winkler_95"] = float(np.mean((ls - li) + 40 * (li - y) * (y < li) + 40 * (y - ls) * (y > ls)))
    return r


def wape(y, previsto) -> float:
    y, previsto = np.asarray(y, float), np.asarray(previsto, float)
    return float(np.sum(np.abs(previsto - y)) / np.sum(np.abs(y)) * 100)


def piso_ruido(y_demanda: pd.Series, tickets: np.ndarray | None, inicio, fim, alvo: str = "demanda",
               simulacoes: int = 1000, semente: int = 2024) -> tuple[float, float]:
    """WAPE mínimo de um modelo de média condicional, se a demanda fosse Poisson(λ_t).

    λ_t = média móvel centrada de 7 dias da demanda observada (oráculo do nível local). Na receita,
    cada transação simulada recebe um ticket sorteado da distribuição empírica. É uma referência
    aproximada e conservadora: um ajuste dentro da amostra **abaixo** dela indica sobreajuste.

    Returns:
        (MAE do piso, WAPE % do piso).
    """
    lam = y_demanda.rolling(7, center=True, min_periods=4).mean().loc[inicio:fim].to_numpy()
    rng = np.random.default_rng(semente)
    cont = rng.poisson(np.tile(lam, (simulacoes, 1)))
    if alvo == "demanda":
        sim, esp = cont, lam
    else:
        sorteio = rng.choice(tickets, size=int(cont.sum()))
        cel = np.repeat(np.arange(cont.size), cont.ravel())
        sim = np.bincount(cel, weights=sorteio, minlength=cont.size).reshape(cont.shape)
        esp = lam * tickets.mean()
    mae = float(np.abs(sim - esp).mean())
    return mae, 100 * mae / float(esp.mean())


def classificar_intermitencia(y: pd.Series) -> dict:
    """Syntetos & Boylan (2005): ADI e CV² das demandas positivas (justifica não usar Croston)."""
    pos = y[y > 0]
    adi = len(y) / max(len(pos), 1)
    cv2 = (pos.std() / pos.mean()) ** 2 if len(pos) > 1 else np.nan
    classe = ("suave" if cv2 < 0.49 else "errática") if adi < 1.32 else ("intermitente" if cv2 < 0.49 else "irregular")
    return {"ADI": adi, "CV²": cv2, "classe": classe}
