"""Tendência, sazonalidade, ciclo e autocorrelação das séries diárias.

Este módulo reúne quatro diagnósticos que fundamentam as escolhas de modelagem (notebook 05):

1. **Tendência.** O teste de Mann-Kendall com a correção de Hamed-Rao para autocorrelação é
   aplicado na série inteira e dentro de cada regime. Em seguida, regressões com dia da semana
   comparam três formas de tendência por AIC/BIC: linear, degraus de nível nas quebras e as duas
   juntas. A pergunta é se o crescimento observado é uma **tendência determinística** ou
   uma **sequência de mudanças de patamar**.
2. **Sazonalidade.** Aplica a metodologia do periodograma com o **teste g exato de Fisher**
   (portada de ``teste.ipynb``), em etapas iterativas:
   - remover a tendência polinomial;
   - calcular o periodograma;
   - testar o maior pico com g = max I(λ_p) / Σ I(λ_p);
   - identificar o período fundamental, que pode ter um harmônico como pico, e retirar o perfil sazonal;
   - repetir até o pico deixar de ser significativo.
3. **Ciclo.** Oscilações mais longas que a semana e sem período fixo. A série é dessazonalizada
   (STL), agregada por semana e tem o nível do regime removido. O desvio restante é suavizado
   pelo filtro HP e testado pelo mesmo teste de Fisher na escala semanal.
4. **Autocorrelação e estacionariedade.** ACF e PACF da série original e das diferenças (Δ1,
   Δ7 e Δ1Δ7), com ADF, KPSS, Zivot-Andrews (raiz unitária com uma quebra) e Ljung-Box.

Todas as funções recebem a série **só até a origem**, para que nenhuma escolha de modelagem
use o período de teste.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import brentq
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf, zivot_andrews

from src.visualization.plots import CORES, DIAS_PT, anotar_quebras, cor_serie, formatar_eixo_datas

ALPHA = 0.05


# ============================================================================ regimes
def rotulos_regime(indice: pd.DatetimeIndex, quebras: list[pd.Timestamp]) -> pd.Series:
    """Número do regime (0, 1, ...) de cada data; cada quebra inicia um novo regime."""
    quebras = sorted(q for q in quebras if q is not None)
    return pd.Series(np.searchsorted(np.array(quebras, dtype="datetime64[ns]"), indice.values, side="right"),
                     index=indice, name="regime")


def nivel_por_regime(y: pd.Series, quebras: list[pd.Timestamp]) -> pd.Series:
    """Tendência em degraus: a média de cada regime repetida em todas as suas datas."""
    return y.groupby(rotulos_regime(y.index, quebras)).transform("mean")


# ============================================================================ 1. tendência
def testes_mann_kendall(y: pd.Series, quebras: list[pd.Timestamp]) -> pd.DataFrame:
    """Mann-Kendall com correção de Hamed-Rao, na série inteira e dentro de cada regime.

    O teste é aplicado às **médias semanais**, o que remove o ciclo de 7 dias sem impor um modelo.
    A correção de Hamed-Rao infla a variância de S pela autocorrelação. Sem ela, o teste rejeita
    "sem tendência" com frequência excessiva em séries persistentes. A inclinação de Sen é
    reportada em unidades por semana.
    """
    semanal = y.resample("W-SUN").mean()
    semanal = semanal[y.resample("W-SUN").size() == 7]
    reg = rotulos_regime(semanal.index - pd.Timedelta(days=3), quebras).values   # semana atribuída pelo dia central
    limites = [y.index.min()] + sorted(q for q in quebras if q is not None) + [y.index.max() + pd.Timedelta(days=1)]
    blocos = [("série inteira", semanal)] + [
        (f"regime {k + 1}: {limites[k]:%d/%m/%y}–{limites[k + 1] - pd.Timedelta(days=1):%d/%m/%y}", semanal[reg == k])
        for k in range(len(limites) - 1)]
    linhas = []
    for rotulo, s in blocos:
        if len(s) < 6:
            linhas.append({"trecho": rotulo, "semanas": len(s), "tendência": "— (poucas semanas)"})
            continue
        r = mk.hamed_rao_modification_test(s.values)
        linhas.append({"trecho": rotulo, "semanas": len(s), "tau de Kendall": r.Tau, "Sen (por semana)": r.slope,
                       "Sen (% da média por semana)": 100 * r.slope / s.mean(), "p (Hamed-Rao)": r.p,
                       "tendência": {"increasing": "crescente", "decreasing": "decrescente"}.get(r.trend, "não significativa")})
    return pd.DataFrame(linhas)


def comparar_formas_tendencia(y: pd.Series, quebras: list[pd.Timestamp], hac_lags: int = 7) -> pd.DataFrame:
    """Compara formas de tendência numa regressão de ``log(1 + y)`` com dia da semana.

    Modelos:
    - sem tendência: só o dia da semana;
    - linear: dia da semana + t;
    - degraus: dia da semana + regime;
    - degraus + linear: dia da semana + regime + t;
    - degraus + linear por regime: inclinação própria em cada regime.

    O erro-padrão da inclinação é HAC (Newey-West), porque os resíduos são autocorrelacionados.
    Um ΔBIC grande a favor dos degraus, com t não significativo dentro deles, indica patamares, e não
    uma tendência determinística.
    """
    ly = np.log1p(y.astype(float))
    t = pd.Series(np.arange(len(y)) / 7.0, index=y.index, name="t (semanas)")
    dow = pd.get_dummies(y.index.dayofweek, prefix="dow", drop_first=True, dtype=float).set_index(y.index)
    reg = rotulos_regime(y.index, quebras)
    degraus = pd.get_dummies(reg, prefix="regime", drop_first=True, dtype=float)
    t_por_regime = pd.DataFrame({f"t × regime {k + 1}": t * (reg == k) for k in sorted(reg.unique())})
    especificacoes = {
        "sem tendência": [dow],
        "linear": [dow, t],
        "degraus (quebras)": [dow, degraus],
        "degraus + linear": [dow, degraus, t],
        "degraus + linear por regime": [dow, degraus, t_por_regime],
    }
    linhas = []
    for nome, partes in especificacoes.items():
        X = sm.add_constant(pd.concat(partes, axis=1))
        m = sm.OLS(ly, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        linha = {"modelo": nome, "parâmetros": int(m.df_model + 1), "AIC": m.aic, "BIC": m.bic, "R²": m.rsquared}
        if "t (semanas)" in X:
            linha["inclinação (% por semana)"] = 100 * np.expm1(m.params["t (semanas)"])
            linha["p da inclinação (HAC)"] = m.pvalues["t (semanas)"]
        if nome == "degraus + linear por regime":
            p = m.pvalues.filter(like="t × regime")
            linha["regimes com inclinação significativa (5%)"] = f"{int((p < ALPHA).sum())} de {len(p)}"
        linhas.append(linha)
    tab = pd.DataFrame(linhas)
    tab["ΔAIC"] = tab["AIC"] - tab["AIC"].min()
    tab["ΔBIC"] = tab["BIC"] - tab["BIC"].min()
    return tab


def inclinacoes_por_regime(y: pd.Series, quebras: list[pd.Timestamp], hac_lags: int = 7) -> pd.DataFrame:
    """Inclinação de ``log(1 + y)`` dentro de cada regime (com dia da semana), erro HAC."""
    limites = [y.index.min()] + sorted(q for q in quebras if q is not None) + [y.index.max() + pd.Timedelta(days=1)]
    linhas = []
    for k in range(len(limites) - 1):
        s = y.loc[limites[k]:limites[k + 1] - pd.Timedelta(days=1)]
        X = pd.get_dummies(s.index.dayofweek, prefix="dow", drop_first=True, dtype=float).set_index(s.index)
        X["t (semanas)"] = np.arange(len(s)) / 7.0
        m = sm.OLS(np.log1p(s.astype(float)), sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        b, ep = m.params["t (semanas)"], m.bse["t (semanas)"]
        linhas.append({"regime": f"{k + 1}: {limites[k]:%d/%m/%y}–{s.index.max():%d/%m/%y}", "dias": len(s),
                       "média diária": s.mean(), "inclinação (% por semana)": 100 * np.expm1(b),
                       "IC 95% inf.": 100 * np.expm1(b - 1.96 * ep), "IC 95% sup.": 100 * np.expm1(b + 1.96 * ep),
                       "p (HAC)": m.pvalues["t (semanas)"],
                       "variação implícita no regime (%)": 100 * np.expm1(b * (len(s) - 1) / 7)})
    return pd.DataFrame(linhas)


def grafico_tendencia(y: pd.Series, quebras: dict[str, pd.Timestamp], nome: str, unidade: str) -> plt.Figure:
    """Série diária, média de 7 dias, tendência em degraus e tendência linear ajustada."""
    datas = [d for d in quebras.values() if d is not None]
    nivel = nivel_por_regime(y, datas)
    t = np.arange(len(y))
    linear = np.polyval(np.polyfit(t, y.values.astype(float), 1), t)
    fig, ax = plt.subplots(figsize=(15, 4.4))
    ax.plot(y.index, y.values, color=CORES["contexto"], lw=0.7, label="Diário")
    ax.plot(y.index, y.rolling(7, min_periods=7).mean(), color=cor_serie(nome), lw=2, label="Média móvel de 7 dias")
    ax.plot(y.index, nivel, color=CORES["observado"], lw=2, drawstyle="steps-post", label="Média do regime (degraus)")
    ax.plot(y.index, linear, color=CORES["observado"], lw=1.4, ls=":", label="Tendência linear (MQO)")
    anotar_quebras(ax, quebras)
    ax.set_ylabel(unidade)
    ax.set_title(f"{nome}: degraus de nível × tendência linear (treino)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    formatar_eixo_datas(ax)
    fig.tight_layout()
    return fig


# ============================================================================ 2. sazonalidade: periodograma + teste g de Fisher
def fisher_pvalor(g: float, n: int) -> float:
    """P[g > z] exato de Fisher: Σ_{j=1}^{a} (-1)^(j-1) C(n, j) (1 - j z)^(n-1), com a = maior inteiro < 1/z.

    A conta usa frações exatas, porque a soma alternada perde precisão em ponto flutuante.
    """
    z = Fraction(float(g))
    a = min(n, floor(1 / z) if (1 / z).denominator != 1 else int(1 / z) - 1)
    soma = sum((-1) ** (j - 1) * comb(n, j) * (1 - j * z) ** (n - 1) for j in range(1, a + 1))
    return float(min(max(soma, 0), 1))


def fisher_critico(n: int, alpha: float = ALPHA) -> float:
    """Valor crítico g_α, tal que P[g > g_α] = α."""
    return brentq(lambda z: fisher_pvalor(z, n) - alpha, 1 / n + 1e-9, 1 - 1e-9)


def periodograma(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """I(λ_p) = |Σ x_t e^{-iλ_p t}|² / T nas frequências de Fourier λ_p = 2πp/T, com p = 1..[(T-1)/2].

    Returns:
        (frequências em ciclos por unidade de tempo, ordenadas do periodograma).
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    F = np.fft.rfft(x - x.mean())
    p = np.arange(1, (T - 1) // 2 + 1)
    return p / T, np.abs(F[p]) ** 2 / T


def ajusta_polinomio(x: np.ndarray, grau: int) -> np.ndarray:
    """Tendência polinomial de grau ``grau``, com t padronizado para estabilidade numérica."""
    t = np.arange(len(x))
    tt = (t - t.mean()) / t.std()
    return np.polyval(np.polyfit(tt, x, grau), tt)


def remove_sazonalidade(x: np.ndarray, P: int) -> tuple[np.ndarray, pd.Series]:
    """Remove o perfil de período P (média por posição no ciclo), o que retira todos os harmônicos de P."""
    pos = np.arange(len(x)) % P
    s = pd.Series(np.asarray(x, dtype=float))
    return s.values - s.groupby(pos).transform("mean").values, s.groupby(pos).mean()


def periodo_fundamental(x: np.ndarray, f: float, k_max: int = 4) -> tuple[int, int]:
    """Período inteiro m que explica o pico na frequência f.

    O pico pode ser um harmônico (f ≈ k/m; 3,5 dias é o harmônico 2 de 7). Além disso, a grade de
    Fourier tem resolução 1/T, e o período verdadeiro pode cair entre duas frequências. A função
    testa todo m compatível (|k/m − f| ≤ 1/T, k = 1..k_max) e escolhe pelo AIC da remoção do perfil.
    """
    T = len(x)
    cands: dict[int, int] = {}
    for k in range(1, k_max + 1):
        lo, hi = k / (f + 1 / T), k / max(f - 1 / T, 1e-9)
        for m in range(max(2, int(np.ceil(lo))), int(np.floor(hi)) + 1):
            if m <= T // 2:
                cands.setdefault(m, k)
    if not cands:
        return int(round(1 / f)), 1

    def aic(m: int) -> float:
        r, _ = remove_sazonalidade(x, m)
        return T * np.log(np.sum(r ** 2) / T) + 2 * m

    m = min(cands, key=aic)
    return m, cands[m]


@dataclass
class ResultadoFisher:
    """Saída da metodologia iterativa do periodograma."""
    nome: str
    serie: pd.Series
    usar_log: bool
    amplitude_media: pd.DataFrame
    regressao_amplitude: dict
    tabela: pd.DataFrame
    perfis: dict[int, pd.Series]
    removidos: list[int]
    etapas: list[dict] = field(default_factory=list)
    unidade_tempo: str = "dias"


def metodologia_fisher(serie: pd.Series, nome: str, alpha: float = ALPHA, grau_inicial: int = 5,
                       grau_iteracoes: int = 2, max_periodos: int = 3, janela_amp: int = 7,
                       tendencia: np.ndarray | None = None, unidade_tempo: str = "dias",
                       decidir_log: bool = True) -> ResultadoFisher:
    """Metodologia do periodograma com o teste g de Fisher (igual a ``teste.ipynb``).

    Etapas:
    1. Testa se a amplitude cresce com a média, regredindo a amplitude de janelas de 7 dias na
       média delas. Se cresce, usa ``log(1 + y)``.
    2. Em cada iteração: remove a tendência (polinômio de grau ``grau_inicial`` na 1ª etapa e
       ``grau_iteracoes`` nas seguintes), calcula o periodograma e testa o maior pico pelo g de
       Fisher. Se o pico é significativo, identifica o período fundamental e retira o perfil sazonal.
    3. Para quando o pico deixa de ser significativo ou o período se repete.

    Args:
        tendencia: tendência alternativa já na escala de trabalho (ex.: degraus de regime). Quando
            dada, substitui o polinômio em todas as etapas. Serve de teste de robustez.
        decidir_log: se False, não aplica o teste amplitude × média e usa a série original.
    """
    y = serie.astype(float)
    grp = np.arange(len(y)) // janela_amp
    am = y.groupby(grp).agg(media="mean", amplitude=lambda s: s.max() - s.min())
    am = am[y.groupby(grp).size().values == janela_amp]
    reg = stats.linregress(am["media"], am["amplitude"])
    usar_log = bool(decidir_log and reg.slope > 0 and reg.pvalue < alpha)
    x = np.log1p(y.values) if usar_log else y.values.copy()

    resultados, perfis, removidos, etapas = [], {}, [], []
    for etapa in range(1, max_periodos + 2):
        grau = grau_inicial if etapa == 1 else grau_iteracoes
        tend = tendencia if tendencia is not None else ajusta_polinomio(x, grau)
        resid = x - tend
        freq, I = periodograma(resid)
        n = len(I)
        k = int(I.argmax())
        g = I[k] / I.sum()
        p_val = fisher_pvalor(g, n)
        g_crit = fisher_critico(n, alpha)
        P, harm = periodo_fundamental(resid, freq[k])
        significativo = g > g_crit
        decisao = "rejeita H0: há componente periódica" if significativo else "não rejeita H0: sem periodicidade"
        etapas.append({"etapa": etapa, "grau": grau, "x": x.copy(), "tendencia": tend, "resid": resid, "freq": freq,
                       "I": I, "k": k, "g": g, "g_crit": g_crit, "P": P, "removidos": list(removidos)})
        resultados.append({"etapa": etapa, "tendência": "degraus" if tendencia is not None else f"polinômio grau {grau}",
                           "removidos antes": str(removidos) if removidos else "—", "freq. do pico": freq[k],
                           f"período do pico ({unidade_tempo})": round(1 / freq[k], 2), "harmônico": harm,
                           "período fundamental": P, "g": g, "g_α": g_crit, "p-valor": p_val, "decisão": decisao})
        if not significativo or etapa > max_periodos or P in removidos or P > len(x) / 2:
            break
        _, perfil = remove_sazonalidade(resid, P)       # perfil estimado na série sem tendência
        perfil = perfil - perfil.mean()
        x = x - perfil.values[np.arange(len(x)) % P]    # retira a sazonalidade da série
        perfis[P] = perfil
        removidos.append(P)

    return ResultadoFisher(nome, serie, usar_log, am, {"inclinação": reg.slope, "p": reg.pvalue, "r": reg.rvalue},
                           pd.DataFrame(resultados), perfis, removidos, etapas, unidade_tempo)


def grafico_amplitude_media(res: ResultadoFisher, cor: str | None = None) -> plt.Figure:
    """Dispersão amplitude × média por janela de 7 dias, com a reta de MQO (decide o log)."""
    am, r = res.amplitude_media, res.regressao_amplitude
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(am["media"], am["amplitude"], facecolors="none", edgecolors=CORES["observado"], s=28)
    xs = np.linspace(am["media"].min(), am["media"].max(), 50)
    b0 = am["amplitude"].mean() - r["inclinação"] * am["media"].mean()
    ax.plot(xs, b0 + r["inclinação"] * xs, color=cor or CORES["Máquina 1"], lw=2)
    ax.set(xlabel="média da janela", ylabel="amplitude (máx − mín)")
    ax.set_title(f"{res.nome}: amplitude × média (janelas de 7 dias)\n"
                 f"inclinação = {r['inclinação']:.2f}, p = {r['p']:.1e} → {'log(1 + y)' if res.usar_log else 'escala original'}")
    fig.tight_layout()
    return fig


def grafico_etapas_fisher(res: ResultadoFisher, cor: str | None = None) -> plt.Figure:
    """Uma linha por etapa: série com a tendência, resíduos e periodograma escalado com o pico testado."""
    cor = cor or CORES["Máquina 1"]
    n = len(res.etapas)
    fig, eixos = plt.subplots(n, 3, figsize=(18, 3.5 * n), squeeze=False)
    rot = "log(1 + y)" if res.usar_log else "y"
    for linha, e in zip(eixos, res.etapas):
        idx = res.serie.index
        linha[0].plot(idx, e["x"], color=CORES["observado"], lw=0.7)
        linha[0].plot(idx, e["tendencia"], color=cor, lw=2)
        tit_rem = f"\n(já sem os períodos {e['removidos']})" if e["removidos"] else ""
        linha[0].set_title(f"Etapa {e['etapa']}: {rot} e tendência ajustada{tit_rem}", fontsize=10.5)
        formatar_eixo_datas(linha[0])
        linha[1].plot(idx, e["resid"], color=CORES["observado"], lw=0.7)
        linha[1].axhline(0, color=CORES["eixo"], lw=1)
        linha[1].set_title("Resíduo (sem tendência)", fontsize=10.5)
        formatar_eixo_datas(linha[1])
        Ie = e["I"] / e["I"].sum()
        if res.unidade_tempo == "dias":           # harmônicos da semana como referência visual
            for h in (1, 2, 3):
                linha[2].axvline(h / 7, color=CORES["grade"], lw=5, zorder=0, label="harmônicos de 7 dias" if h == 1 else None)
        linha[2].plot(e["freq"], Ie, color=CORES["observado"], lw=0.8)
        linha[2].axhline(e["g_crit"], color=CORES["quebra"], ls="--", lw=1, label=f"g_α = {e['g_crit']:.3f}")
        linha[2].scatter(e["freq"][e["k"]], Ie[e["k"]], color=cor, s=45, zorder=5)
        linha[2].annotate(f"pico: {1 / e['freq'][e['k']]:.2f} {res.unidade_tempo}\nperíodo fundamental = {e['P']}",
                          (e["freq"][e["k"]], Ie[e["k"]]), textcoords="offset points", xytext=(8, -14), fontsize=9, color=cor)
        linha[2].set_title(f"Periodograma escalado: g = {e['g']:.3f}", fontsize=10.5)
        linha[2].set_xlabel(f"frequência (ciclos/{res.unidade_tempo[:-1] if res.unidade_tempo.endswith('s') else res.unidade_tempo})")
        linha[2].legend(loc="upper right")
    fig.suptitle(f"{res.nome}: metodologia do periodograma com o teste g de Fisher", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def grafico_perfis(res: ResultadoFisher, cor: str | None = None) -> plt.Figure | None:
    """Perfis sazonais extraídos, um painel por período. O de 7 dias recebe os nomes dos dias."""
    if not res.perfis:
        return None
    cor = cor or CORES["Máquina 1"]
    fig, eixos = plt.subplots(1, len(res.perfis), figsize=(6 * len(res.perfis), 3.6), squeeze=False)
    d0 = res.serie.index[0].weekday()
    for ax, (P, perfil) in zip(eixos[0], res.perfis.items()):
        efeito = np.expm1(perfil.values) * 100 if res.usar_log else perfil.values
        ax.bar(perfil.index, efeito, color=cor, width=0.7)
        ax.axhline(0, color=CORES["eixo"], lw=1)
        if P == 7:
            ax.set_xticks(perfil.index, [DIAS_PT[(d0 + i) % 7] for i in perfil.index])
        ax.set(xlabel="posição no ciclo", ylabel="efeito (%)" if res.usar_log else "efeito (unidades)")
        ax.set_title(f"{res.nome}: perfil sazonal de período {P}")
    fig.tight_layout()
    return fig


def teste_fisher_banda(x: np.ndarray, periodo_min: float = 2.0, periodo_max: float = 31.0,
                       alpha: float = ALPHA) -> dict:
    """Teste g de Fisher restrito a uma faixa de períodos **fixada antes de olhar os dados**.

    Sob H0 (ruído branco), as ordenadas do periodograma são exponenciais i.i.d. Então o g calculado
    sobre um subconjunto escolhido a priori, ``max I / Σ I`` nesse subconjunto, segue a mesma
    distribuição de Fisher com n = tamanho do subconjunto. A faixa de 2 a 31 dias isola a
    sazonalidade intramensal das baixas frequências, que pertencem à tendência e ao ciclo.
    """
    freq, I = periodograma(x)
    banda = (freq >= 1 / periodo_max) & (freq <= 1 / periodo_min)
    fb, Ib = freq[banda], I[banda]
    k = int(Ib.argmax())
    g = Ib[k] / Ib.sum()
    P, harm = periodo_fundamental(x, fb[k])
    p_val = fisher_pvalor(g, len(Ib))
    ordem = np.argsort(Ib)[::-1]                                  # ordenadas da maior para a menor
    harm7 = [int(np.abs(fb - h / 7).argmin()) for h in (1, 2, 3)]   # ordenadas mais próximas de 1/7, 2/7 e 3/7
    ranque_7 = [int(np.flatnonzero(ordem == j)[0]) + 1 for j in harm7]
    return {"faixa (dias)": f"{periodo_min:g}–{periodo_max:g}", "ordenadas": len(Ib), "período do pico": round(1 / fb[k], 2),
            "período fundamental": P, "harmônico": harm, "g": g, "g_α": fisher_critico(len(Ib), alpha), "p-valor": p_val,
            "posição dos harmônicos de 7 no ranque": ranque_7,
            "decisão": "rejeita H0" if p_val < alpha else "não rejeita H0"}


def teste_sazonalidade_semanal(y: pd.Series, quebras: list[pd.Timestamp], min_dias: int = 28) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Teste do **período conhecido** (7 dias) por regime e teste de estabilidade do perfil.

    O teste g de Fisher procura o maior pico entre todas as frequências. Ele não usa a informação
    de que o calendário tem 7 dias, e o padrão semanal reparte a potência entre 3 harmônicos
    (1/7, 2/7 e 3/7). Quando o período é conhecido a priori, o teste mais poderoso é o F das 6
    dummies de dia da semana. Ele equivale a testar a soma do periodograma nos harmônicos de 7.

    Returns:
        (teste F por regime, perfil em % sobre a média do regime, teste de estabilidade regime × dia).
    """
    ly = np.log1p(y.astype(float))
    reg = rotulos_regime(y.index, quebras)
    limites = [y.index.min()] + sorted(q for q in quebras if q is not None) + [y.index.max() + pd.Timedelta(days=1)]
    linhas, perfis = [], {}
    for k in range(len(limites) - 1):
        s = ly[reg == k]
        rot = f"{k + 1}: {limites[k]:%d/%m/%y}–{s.index.max():%d/%m/%y}"
        bruto = y[reg == k]
        perfil = bruto.groupby(bruto.index.dayofweek).mean()
        perfis[rot] = 100 * (perfil / perfil.mean() - 1)
        if len(s) < min_dias:
            continue
        X = sm.add_constant(pd.get_dummies(s.index.dayofweek, prefix="dow", drop_first=True, dtype=float).set_index(s.index))
        m = sm.OLS(s, X).fit()
        f = m.f_test(np.eye(len(m.params))[1:])
        linhas.append({"regime": rot, "dias": len(s), "F (dia da semana)": float(f.fvalue), "p": float(f.pvalue),
                       "R² do dia da semana": m.rsquared, "amplitude do perfil (p.p.)": perfis[rot].max() - perfis[rot].min(),
                       "dia mais forte": DIAS_PT[int(perfil.idxmax())], "dia mais fraco": DIAS_PT[int(perfil.idxmin())]})
    perfis = pd.DataFrame(perfis).T
    perfis.columns = [DIAS_PT[c] for c in perfis.columns]
    # estabilidade: regime + dia × (regime + dia + regime × dia)
    df = pd.DataFrame({"ly": ly, "reg": reg.astype(str), "dow": y.index.dayofweek.astype(str)})
    m0 = sm.formula.ols("ly ~ C(reg) + C(dow)", df).fit()
    m1 = sm.formula.ols("ly ~ C(reg) * C(dow)", df).fit()
    comp = sm.stats.anova_lm(m0, m1)
    # teste dirigido: o perfil do último regime difere do perfil comum dos anteriores? (6 gl em vez de 6 × (R − 1))
    X0 = pd.DataFrame(m0.model.exog, columns=m0.model.exog_names, index=y.index)
    dows = X0.filter(like="C(dow)")
    X2 = pd.concat([X0, dows.mul((reg == reg.max()).astype(float), axis=0).add_suffix(" × último")], axis=1)
    m2 = sm.OLS(ly, X2).fit()
    comp2 = sm.stats.anova_lm(sm.OLS(ly, X0).fit(), m2)
    estabilidade = {"F (interação regime × dia, todos)": comp["F"].iloc[1], "gl": int(comp["df_diff"].iloc[1]),
                    "p (todos)": comp["Pr(>F)"].iloc[1], "ΔBIC (todos)": m1.bic - m0.bic,
                    "F (perfil do último regime ≠ anteriores)": comp2["F"].iloc[1], "gl (último)": int(comp2["df_diff"].iloc[1]),
                    "p (último)": comp2["Pr(>F)"].iloc[1], "ΔBIC (último)": m2.bic - m0.bic}
    return pd.DataFrame(linhas), perfis, estabilidade


def grafico_perfis_semanais(perfis: pd.DataFrame, nome: str) -> plt.Figure:
    """Mapa de calor do perfil semanal de cada regime (% sobre a média do regime)."""
    from src.visualization.plots import DIVERGENTE
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(perfis) + 1.6))
    lim = np.nanmax(np.abs(perfis.values))
    im = ax.imshow(perfis.values, cmap=DIVERGENTE, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(perfis.shape[0]):
        for j in range(perfis.shape[1]):
            v = perfis.iat[i, j]
            ax.text(j, i, f"{v:+.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.6 * lim else CORES["observado"])
    ax.set_xticks(range(perfis.shape[1]), perfis.columns)
    ax.set_yticks(range(perfis.shape[0]), [f"regime {r}" for r in perfis.index])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="% sobre a média do regime", shrink=0.8)
    ax.set_title(f"{nome}: perfil semanal da demanda em cada regime")
    fig.tight_layout()
    return fig


def forca_componentes(y: pd.Series, periodo: int = 7) -> dict:
    """Força da tendência e da sazonalidade pela STL robusta (Wang, Smith e Hyndman, 2006).

    F = max(0, 1 − Var(R) / Var(C + R)), em que C é o componente avaliado. F ≈ 0 significa que o
    componente não se destaca do ruído, e F ≈ 1 que domina a variação.
    """
    r = STL(y.astype(float), period=periodo, robust=True).fit()
    vr = np.var(r.resid)
    return {"força da tendência": max(0.0, 1 - vr / np.var(r.trend + r.resid)),
            "força da sazonalidade": max(0.0, 1 - vr / np.var(r.seasonal + r.resid))}


# ============================================================================ 3. ciclo
def lambda_hp_corte(periodo: float) -> float:
    """λ do filtro HP cujo ganho é 1/2 no período ``periodo``: λ = 1 / [4 (1 − cos(2π / P))²].

    A parte suave do HP retém as oscilações mais longas que P. Com P = 13 semanas, o "ciclo" é o
    movimento com duração de um trimestre ou mais. A regra de Ravn-Uhlig para dados semanais
    (λ ≈ 4,6 × 10⁷) mira ciclos de negócios de anos e, com 53 semanas, só produziria uma reta.
    """
    return 1.0 / (4 * (1 - np.cos(2 * np.pi / periodo)) ** 2)


def componente_ciclo(y: pd.Series, quebras: list[pd.Timestamp], periodo_corte: float = 13) -> pd.DataFrame:
    """Separa o ciclo da tendência em escala semanal.

    Passos:
    1. ``log(1 + y)`` dessazonalizado pela STL robusta de período 7.
    2. Média por semana completa (seg–dom).
    3. Subtração do nível do regime (degraus). O resultado é o **desvio**, que contém ciclo e ruído.
    4. Filtro HP sobre o desvio (λ com corte em ``periodo_corte`` semanas). A parte suave é o **ciclo**.

    Com isso, os degraus não são confundidos com ciclo, e o HP não "alisa" as quebras.
    """
    ly = np.log1p(y.astype(float))
    stl = STL(ly, period=7, robust=True).fit()
    dessaz = ly - stl.seasonal
    tam = dessaz.resample("W-SUN").size()
    semanal = dessaz.resample("W-SUN").mean()[tam == 7]
    meio_semana = semanal.index - pd.Timedelta(days=3)          # semana atribuída ao regime do dia central
    nivel = semanal.groupby(rotulos_regime(meio_semana, quebras).values).transform("mean")
    desvio = semanal - nivel
    _, ciclo = hpfilter(desvio, lamb=lambda_hp_corte(periodo_corte))
    # hpfilter devolve (ciclo, tendência); aqui a "tendência" do HP do desvio é o ciclo de baixa frequência
    return pd.DataFrame({"dessazonalizada (log)": semanal, "nível do regime": nivel, "desvio": desvio,
                         "ciclo (HP)": ciclo, "irregular": desvio - ciclo})


def resumo_ciclo(c: pd.DataFrame) -> dict:
    """Amplitude do ciclo, duração média entre picos e parte da variância do desvio que ele explica."""
    ciclo = c["ciclo (HP)"]
    sinal = np.sign(ciclo.values)
    trocas = np.flatnonzero(np.diff(sinal) != 0)
    duracao = np.diff(trocas).mean() * 2 if len(trocas) > 1 else np.nan
    lb = acorr_ljungbox(c["desvio"], lags=[4, 8, 13], return_df=True)["lb_pvalue"]
    return {"semanas": len(c), "Ljung-Box p do desvio (4 / 8 / 13 semanas)": " / ".join(f"{v:.2f}" for v in lb), "amplitude do ciclo (% do nível, ±)": 100 * np.expm1(ciclo.abs().max()),
            "desvio-padrão do ciclo (%)": 100 * np.expm1(ciclo.std()),
            "fração da variância do desvio explicada pelo ciclo": ciclo.var() / c["desvio"].var(),
            "duração média de um ciclo completo (semanas)": duracao,
            "trocas de sinal": len(trocas)}


def grafico_ciclo(c: pd.DataFrame, quebras: dict[str, pd.Timestamp], nome: str) -> plt.Figure:
    """Painel superior: série semanal dessazonalizada com o nível do regime. Inferior: desvio e ciclo HP."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 6.4), sharex=True)
    a1.plot(c.index, c["dessazonalizada (log)"], color=CORES["observado"], lw=1.4, marker="o", ms=3,
            label="Média semanal dessazonalizada")
    a1.plot(c.index, c["nível do regime"], color=cor_serie(nome), lw=2, drawstyle="steps-mid", label="Nível do regime")
    anotar_quebras(a1, quebras)
    a1.set_ylabel("log(1 + transações/dia)")
    a1.set_title(f"{nome}: tendência em degraus")
    a1.legend(loc="lower right")
    a2.bar(c.index, c["desvio"], width=5, color=CORES["contexto"], label="Desvio do nível (ciclo + ruído)")
    a2.plot(c.index, c["ciclo (HP)"], color=cor_serie(nome), lw=2.2, label="Ciclo (filtro HP)")
    a2.axhline(0, color=CORES["eixo"], lw=1)
    a2.set_ylabel("desvio em log")
    a2.set_title("Ciclo: desvio semanal em relação ao nível do regime")
    a2.legend(loc="lower right")
    formatar_eixo_datas(a2)
    fig.tight_layout()
    return fig


def efeito_dia_do_mes(y: pd.Series, quebras: list[pd.Timestamp]) -> pd.DataFrame:
    """Testa um ciclo mensal (pagamento, início do mês) no resíduo dessazonalizado e sem degraus.

    Compara os terços do mês (dias 1–10, 11–20 e 21–31) por Kruskal-Wallis, que não supõe normalidade.
    O efeito é medido em % sobre o nível.
    """
    ly = np.log1p(y.astype(float))
    stl = STL(ly, period=7, robust=True).fit()
    r = ly - stl.seasonal
    r = r - nivel_por_regime(r, quebras)
    terco = pd.cut(r.index.day, [0, 10, 20, 31], labels=["dias 1–10", "dias 11–20", "dias 21–31"])
    grupos = [r[terco == g].values for g in terco.categories]
    h, p = stats.kruskal(*grupos)
    tab = r.groupby(terco, observed=True).agg(dias="size", efeito_log="mean")
    tab["efeito (% sobre o nível)"] = 100 * np.expm1(tab.pop("efeito_log"))
    tab.attrs.update({"H de Kruskal-Wallis": h, "p": p})
    return tab


# ============================================================================ 4. autocorrelação e estacionariedade
def transformacoes_diferenca(y: pd.Series, quebras: list[pd.Timestamp] | None = None) -> dict[str, pd.Series]:
    """Série original e as diferenças: Δ1 (tendência), Δ7 (sazonal) e Δ1Δ7 (as duas).

    Com ``quebras``, inclui a alternativa à diferença: subtrair o nível de cada regime (degraus).
    """
    y = y.astype(float)
    t = {"original": y, "Δ1 (1ª diferença)": y.diff().dropna(), "Δ7 (diferença sazonal)": y.diff(7).dropna(),
         "Δ1Δ7": y.diff(7).diff().dropna()}
    if quebras:
        sem = y - nivel_por_regime(y, quebras)
        t["sem degraus (y − nível do regime)"] = sem
        t["sem degraus, Δ1"] = sem.diff().dropna()
    return t


def testes_estacionariedade(y: pd.Series, nome: str = "", quebras: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    """ADF, KPSS e Ljung-Box para cada transformação, mais Zivot-Andrews para a série original.

    As hipóteses dos testes são opostas:
    - **ADF**: H0 = raiz unitária;
    - **KPSS**: H0 = estacionária.
    Uma série é estacionária quando o ADF rejeita e o KPSS não rejeita. Na série original, o teste
    de **Zivot-Andrews** permite uma quebra no intercepto. Ele evita o viés de Perron (1989): uma
    série estacionária com mudança de nível parece ter raiz unitária para o ADF.
    """
    linhas = []
    for rotulo, s in transformacoes_diferenca(y, quebras).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = adfuller(s, autolag="AIC", regression="c")
            kp = kpss(s, regression="c", nlags="auto")
        lb = acorr_ljungbox(s, lags=[7, 14], return_df=True)["lb_pvalue"]
        linha = {"série": nome, "transformação": rotulo, "n": len(s), "ADF estat.": adf[0], "ADF p": adf[1],
                 "KPSS estat.": kp[0], "KPSS p": kp[1], "Ljung-Box p (7)": lb.loc[7], "Ljung-Box p (14)": lb.loc[14]}
        linha["diagnóstico"] = ("estacionária" if adf[1] < ALPHA and kp[1] >= ALPHA else
                                "não estacionária" if adf[1] >= ALPHA and kp[1] < ALPHA else "inconclusivo (testes divergem)")
        if rotulo == "original":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                za = zivot_andrews(s, regression="c", autolag="AIC")
            linha.update({"Zivot-Andrews estat.": za[0], "Zivot-Andrews p": za[1], "quebra ZA": s.index[za[4]]})
        linhas.append(linha)
    return pd.DataFrame(linhas)


def tabela_acf_pacf(y: pd.Series, lags: tuple[int, ...] = (1, 2, 3, 6, 7, 8, 14, 21, 28),
                    quebras: list[pd.Timestamp] | None = None) -> pd.DataFrame:
    """ACF e PACF em defasagens selecionadas para cada transformação, com a banda de ±1,96/√n."""
    linhas = []
    for rotulo, s in transformacoes_diferenca(y, quebras).items():
        a = acf(s, nlags=max(lags), fft=True)
        p = pacf(s, nlags=max(lags), method="ywm")
        for tipo, v in [("ACF", a), ("PACF", p)]:
            linhas.append({"transformação": rotulo, "função": tipo, **{f"lag {k}": v[k] for k in lags},
                           "banda ±": 1.96 / np.sqrt(len(s))})
    return pd.DataFrame(linhas)


def grafico_acf_pacf(y: pd.Series, nome: str, nlags: int = 35,
                     transformacoes: tuple[str, ...] = ("original", "Δ1 (1ª diferença)", "Δ7 (diferença sazonal)", "Δ1Δ7"),
                     quebras: list[pd.Timestamp] | None = None, titulo: str | None = None) -> plt.Figure:
    """Série, ACF e PACF de cada transformação, com as defasagens múltiplas de 7 destacadas."""
    series = transformacoes_diferenca(y, quebras)
    cor = cor_serie(nome)
    fig, eixos = plt.subplots(len(transformacoes), 3, figsize=(18, 3.3 * len(transformacoes)), squeeze=False,
                              gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for linha, rotulo in zip(eixos, transformacoes):
        s = series[rotulo]
        linha[0].plot(s.index, s.values, color=CORES["observado"], lw=0.7)
        linha[0].axhline(0 if rotulo != "original" else s.mean(), color=CORES["eixo"], lw=1)
        linha[0].set_title(f"{rotulo}", fontsize=10.5)
        formatar_eixo_datas(linha[0])
        for ax, f, tit in [(linha[1], plot_acf, "ACF"), (linha[2], plot_pacf, "PACF")]:
            kw = {"method": "ywm"} if f is plot_pacf else {}
            f(s, lags=nlags, ax=ax, zero=False, color=cor, vlines_kwargs={"colors": cor}, alpha=ALPHA, **kw)
            for k in range(7, nlags + 1, 7):
                ax.axvline(k, color=CORES["grade"], lw=6, zorder=0)
            ax.set_title(f"{tit}: {rotulo}", fontsize=10.5)
            ax.set_xlabel("defasagem (dias); faixas = múltiplos de 7")
            ax.set_ylim(-1.05, 1.05)
    fig.suptitle(titulo or f"{nome}: série, ACF e PACF antes e depois da diferenciação", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def perfil_semanal_antes_depois(y: pd.Series, quebras: list[pd.Timestamp], corte: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perfil semanal antes e depois de ``corte`` (% sobre a média), descontando o nível de cada regime.

    Antes do corte há vários patamares. Cada dia é dividido pela média do seu regime antes de
    tirar a média por dia da semana, para que uma mudança de nível não pareça padrão semanal.
    O teste F das 6 dummies de dia da semana (em log1p) inclui efeitos fixos de regime.

    Returns:
        (perfil: períodos × dias da semana em %, teste por período).
    """
    reg = rotulos_regime(y.index, quebras)
    relativo = y / y.groupby(reg).transform("mean")
    perfis, testes = {}, []
    for nome, msk in [(f"antes de {corte:%d/%m/%Y}", y.index < corte), (f"desde {corte:%d/%m/%Y}", y.index >= corte)]:
        if msk.sum() < 14:
            continue
        r = relativo[msk]
        perfis[nome] = 100 * (r.groupby(r.index.dayofweek).mean() - 1)
        df = pd.DataFrame({"ly": np.log1p(y[msk].astype(float)), "reg": reg[msk].astype(str), "dow": y.index[msk].dayofweek.astype(str)})
        efeitos = "C(reg) + " if df["reg"].nunique() > 1 else ""
        m1 = sm.formula.ols(f"ly ~ {efeitos}C(dow)", df).fit()
        m0 = sm.formula.ols(f"ly ~ {efeitos.rstrip(' + ') or '1'}", df).fit()
        comp = sm.stats.anova_lm(m0, m1)
        testes.append({"período": nome, "dias": int(msk.sum()), "F (dia da semana | regime)": comp["F"].iloc[1],
                       "p": comp["Pr(>F)"].iloc[1], "R² parcial do dia da semana": 1 - m1.ssr / m0.ssr,
                       "fim de semana vs dias úteis (%)": 100 * (r[r.index.dayofweek >= 5].mean() / r[r.index.dayofweek < 5].mean() - 1)})
    perfis = pd.DataFrame(perfis).T
    perfis.columns = [DIAS_PT[c] for c in perfis.columns]
    return perfis, pd.DataFrame(testes)


def grafico_perfil_antes_depois(perfis: pd.DataFrame, nome: str, cor: str | None = None) -> plt.Figure:
    """Barras agrupadas do perfil semanal antes × depois do corte (% sobre a média do regime)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(perfis.shape[1])
    larg = 0.38
    for k, (periodo, linha) in enumerate(perfis.iterrows()):
        c = CORES["contexto"] if k == 0 else (cor or cor_serie("Máquina 1"))
        barras = ax.bar(x + (k - 0.5) * larg, linha.values, width=larg - 0.02, color=c, label=periodo)
        if k == perfis.shape[0] - 1:
            ax.bar_label(barras, labels=[f"{v:+.0f}%" for v in linha.values], fontsize=8, padding=2, color=CORES["texto_secundario"])
    ax.axhline(0, color=CORES["eixo"], lw=1)
    ax.set_xticks(x, perfis.columns)
    ax.set_ylabel("% sobre a média do regime")
    ax.set_title(f"{nome}: padrão semanal só aparece no regime de 2025")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


# ============================================================================ síntese para relatórios
def resumo_estrutura_temporal(y: pd.Series, quebras: list[pd.Timestamp], nome: str = "") -> dict:
    """Uma linha com os diagnósticos do notebook 05 que levam a decisões (tendência, sazonalidade, ciclo, mês).

    Recebe a série só até a origem e as quebras detectadas com o treino (inclui a ampliação do horário).
    """
    formas = comparar_formas_tendencia(y, quebras).set_index("modelo")
    incl = inclinacoes_por_regime(y, quebras)
    ly = pd.Series(np.log1p(y.values.astype(float)), index=y.index)
    nivel = nivel_por_regime(ly, quebras).values
    fisher_ano = metodologia_fisher(y, nome)
    fisher_degraus = metodologia_fisher(y, nome, tendencia=nivel)
    banda = teste_fisher_banda(ly.values - nivel)
    c = componente_ciclo(y, quebras)
    rc = resumo_ciclo(c)
    rf = metodologia_fisher(c["desvio"], f"{nome} — desvio semanal", grau_inicial=0, grau_iteracoes=0,
                            decidir_log=False, unidade_tempo="semanas", janela_amp=4)
    dm = efeito_dia_do_mes(y, quebras)
    return {"série": nome, "dias": len(y), "fim": y.index.max(),
            "inclinação no ano inteiro (% por semana)": formas.loc["linear", "inclinação (% por semana)"],
            "ΔBIC reta − degraus": formas.loc["linear", "BIC"] - formas.loc["degraus (quebras)", "BIC"],
            "regimes com inclinação significativa": int((incl["p (HAC)"] < ALPHA).sum()), "regimes": len(incl),
            "Fisher no ano inteiro: períodos dos picos (dias)": " / ".join(f"{v:.0f}" for v in fisher_ano.tabela["período do pico (dias)"].head(3)),
            "Fisher com degraus: p": float(fisher_degraus.tabela.iloc[0]["p-valor"]),
            "Fisher 2–31 dias após degraus: p": float(banda["p-valor"]),
            "ciclo: Fisher no desvio semanal (p)": float(rf.tabela.iloc[0]["p-valor"]),
            "ciclo: Ljung-Box p mínimo": min(float(v) for v in rc["Ljung-Box p do desvio (4 / 8 / 13 semanas)"].split(" / ")),
            "dia do mês: p Kruskal-Wallis": float(dm.attrs["p"])}


def serie_degraus_semanal(y: pd.Series, quebras: list[pd.Timestamp]) -> pd.DataFrame:
    """Média semanal da demanda, nível do regime e reta ajustada (para comparar degraus × tendência num gráfico)."""
    t = np.arange(len(y), dtype=float)
    b, a = np.polyfit(t, y.astype(float).values, 1)
    d = pd.DataFrame({"demanda": y.astype(float), "nível do regime": nivel_por_regime(y.astype(float), quebras),
                      "reta": a + b * t}, index=y.index)
    return d.resample("W").mean()


# ============================================================================ metodologia de teste.ipynb só em 2025
# Recortes de 2025 para a metodologia de Fisher. ``teste.ipynb`` usa DATA_INICIO = 01/03/2025 e sugere
# 01/02/2025 "para incluir fevereiro"; a quebra de 2025 e o ano-calendário entram como sensibilidade.
RECORTES_2025 = {"desde 01/02/2025 (opção do teste.ipynb)": "2025-02-01",
                 "desde 30/01/2025 (quebra de 2025)": "2025-01-30",
                 "desde 01/01/2025 (ano-calendário)": "2025-01-01",
                 "desde 01/03/2025 (padrão do teste.ipynb)": "2025-03-01"}


def fisher_recortes(series: dict[tuple[str, str], pd.Series], chaves: list[tuple[str, str]],
                    recortes: dict[str, str] = RECORTES_2025, fins: tuple[pd.Timestamp | None, ...] = (None,)) -> pd.DataFrame:
    """Aplica ``metodologia_fisher`` (igual a ``teste.ipynb``) em cada recorte de 2025 e resume a 1ª etapa.

    Args:
        series: ``{(série, alvo): pd.Series}`` diárias.
        chaves: pares (série, alvo) a testar.
        recortes: ``{rótulo: data de início}``.
        fins: últimas datas a usar (``None`` = série inteira, como em ``teste.ipynb``).
    """
    linhas = []
    for fim in fins:
        for (serie, alvo) in chaves:
            for rotulo, inicio in recortes.items():
                y = series[(serie, alvo)].loc[inicio:fim].dropna()
                if len(y) < 14:
                    continue
                r = metodologia_fisher(y, f"{serie} — {alvo} ({rotulo})")
                e = r.tabela.iloc[0]
                linhas.append({"série": serie, "alvo": alvo, "recorte": rotulo, "início": y.index.min(), "fim": y.index.max(),
                               "dias": len(y), "escala": "log(1 + y)" if r.usar_log else "original",
                               "pico da 1ª etapa (dias)": e["período do pico (dias)"], "período fundamental": int(e["período fundamental"]),
                               "g": float(e["g"]), "g_α (5%)": float(e["g_α"]), "p-valor": float(e["p-valor"]),
                               "periodicidades encontradas": ", ".join(map(str, r.removidos)) or "nenhuma",
                               "período 7 significativo": 7 in r.removidos})
    return pd.DataFrame(linhas)


def periodograma_etapa(res: ResultadoFisher, etapa: int = 1) -> pd.DataFrame:
    """Periodograma escalado (I / ΣI) de uma etapa, com o limite g_α e o pico marcado."""
    e = res.etapas[etapa - 1]
    escala = e["I"] / e["I"].sum()
    return pd.DataFrame({"frequência (ciclos/dia)": e["freq"], "período (dias)": 1 / e["freq"], "ordenada escalada": escala,
                         "g_α (5%)": e["g_crit"], "pico": np.arange(len(escala)) == e["k"]})


def perfil_semanal_fisher(res: ResultadoFisher, periodo: int = 7) -> pd.Series:
    """Perfil do período 7 retirado pela metodologia, rotulado pelo dia da semana (na escala de trabalho)."""
    perfil = res.perfis[periodo]
    d0 = res.serie.index[0].dayofweek
    rot = pd.Series(perfil.values, index=[DIAS_PT[(d0 + i) % 7] for i in perfil.index], name=f"efeito ({'log' if res.usar_log else 'vendas/dia'})")
    return rot.reindex(DIAS_PT)
