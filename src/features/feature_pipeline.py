"""Pipeline de atributos para os modelos de Machine Learning (sem vazamento temporal).

Para um dia-alvo d e uma origem T (último dia observado):

| Grupo | Atributos | Disponibilidade |
|---|---|---|
| Defasagens e janelas | ``lag_1..28``, média/DP/CV de 7, 14 e 30 dias | até d − 1; no horizonte, a recursão usa as próprias previsões |
| Calendário | dia da semana, mês, fim de semana, feriado, Fourier etc. | determinístico |
| Tendência | ``tempo`` (dias desde 01/03/2024) | determinístico |
| Quebras | ``dummy_horario_reduzido``, ``dummy_novos_clientes``, ``regime_*``, ``maquinas_ativas`` | detectadas só com dados até T |
| Comportamentais | média de 7 dias de: % de vendas de manhã e à noite, ticket médio, itens por pedido; na M1, cartões únicos e % de transações de cartões novos | até d − 1; **congeladas no valor de T** para d > T + 1 (não são previstas) |

Dois testes garantem a ausência de vazamento (também em ``tests/test_features.py``):
1. substituir todo o futuro (> T) por ruído não pode alterar nenhum atributo até T + 1;
2. calcular os atributos a partir de uma cauda curta da série dá o mesmo resultado que a série inteira
   (garante que a previsão recursiva, que usa só a cauda, é consistente com o treino).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.preprocessing import BaseDados
from src.features.break_dummies import dummies_quebras
from src.features.lag_features import atributos_defasados, cauda_necessaria
from src.features.temporal_features import features_calendario
from src.settings import carregar_config
from src.structural_breaks.structural_break_test import DetectorQuebras

CATEGORICAS = ["dia_semana", "mes"]


def comportamentais_diarios(transacoes: pd.DataFrame, serie: str) -> pd.DataFrame:
    """Indicadores diários de comportamento de compra (valores do próprio dia, ainda sem defasagem)."""
    cfg = carregar_config()["features"]
    t = transacoes if serie == "Total" else transacoes[transacoes["serie"] == serie]
    g = t.groupby("date")
    d = pd.DataFrame({
        "pct_manha": g["hora"].apply(lambda h: (h <= cfg["manha_ate_hora"]).mean()),
        "pct_noite": g["hora"].apply(lambda h: (h >= cfg["noite_desde_hora"]).mean()),
        "ticket_medio": g["money"].mean(),
        "itens_por_pedido": g.size() / g["datetime"].nunique(),
    })
    if serie == "Máquina 1":
        m1 = t[t["card"].notna()].copy()
        m1["primeira"] = m1.groupby("card")["date"].transform("min")
        gc = m1.groupby("date")
        d["cartoes_unicos"] = gc["card"].nunique()
        d["pct_cartoes_novos"] = (m1["date"] == m1["primeira"]).groupby(m1["date"]).mean()
    return d


@dataclass
class ConstrutorAtributos:
    """Constrói a matriz de atributos de uma série com a informação disponível numa origem.

    Args:
        base: dados do projeto.
        detector: detector de quebras (com cache) para as dummies conhecidas na origem.
        serie: "Máquina 1", "Máquina 2" ou "Total".
        origem: último dia observado (T).
        comportamentais: inclui os atributos comportamentais.
    """
    base: BaseDados
    detector: DetectorQuebras
    serie: str
    origem: pd.Timestamp
    comportamentais: bool = True

    def __post_init__(self):
        cfg = carregar_config()
        self._cfg = cfg
        self._inicio_global = self.base.series[("Total", "demanda")].index.min()
        if self.comportamentais:
            diarios = comportamentais_diarios(self.base.transacoes[self.base.transacoes["date"] <= self.origem], self.serie)
            # Calendário até T + 1: a linha T + 1 recebe a média dos 7 dias até T (último valor conhecido)
            cal = pd.date_range(self.base.inicio[self.serie], self.origem + pd.Timedelta(days=1))
            comp = diarios.reindex(cal)
            if "cartoes_unicos" in comp:
                comp["cartoes_unicos"] = comp["cartoes_unicos"].fillna(0)
            self._comp = comp.rolling(7, min_periods=1).mean().shift(1)

    def __call__(self, y: pd.Series) -> pd.DataFrame:
        """Atributos para cada data de ``y.index`` usando apenas valores anteriores a ela."""
        datas = y.index
        X = atributos_defasados(y)
        X = X.join(features_calendario(datas))
        X["tempo"] = (datas - self._inicio_global).days
        X = X.join(dummies_quebras(datas, self.serie, self.origem, self.detector, self.base.inicio["Máquina 2"]))
        if self.comportamentais:
            # Para d > T + 1 os indicadores não são conhecidos: mantém-se o último valor disponível (linha T + 1)
            comp = self._comp.reindex(datas)
            futuras = datas > self._comp.index.max()
            if futuras.any():
                comp.loc[futuras] = self._comp.iloc[-1].to_numpy()
            X = X.join(comp.add_prefix("comp_"))
        return X


def teste_vazamento_futuro(construtor: ConstrutorAtributos, y_completo: pd.Series, semente: int = 1) -> bool:
    """Substitui o futuro (> origem) por ruído e verifica que os atributos até origem + 1 não mudam."""
    rng = np.random.default_rng(semente)
    y_ruido = y_completo.copy()
    futuro = y_ruido.index > construtor.origem
    y_ruido[futuro] = rng.integers(0, 1000, futuro.sum())
    corte = construtor.origem + pd.Timedelta(days=1)
    a, b = construtor(y_completo).loc[:corte], construtor(y_ruido).loc[:corte]
    return a.equals(b)


def teste_consistencia_recursao(construtor: ConstrutorAtributos, y: pd.Series, n_datas: int = 20) -> bool:
    """Atributos calculados com só a cauda da série são iguais aos calculados com a série inteira."""
    completo = construtor(y)
    cauda = cauda_necessaria()
    for d in completo.index[-n_datas:]:
        parcial = construtor(y.loc[:d].iloc[-cauda:])
        pd.testing.assert_series_equal(parcial.loc[d], completo.loc[d], check_names=False)
    return True


# ================================================================== análise de atributos
def correlacao_com_alvo(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Correlação de Spearman de cada atributo numérico com o alvo (só treino)."""
    from scipy import stats

    saida = {}
    for c in X.columns:
        v = pd.concat([X[c], y], axis=1).dropna()
        if len(v) > 10 and v.iloc[:, 0].nunique() > 1:
            saida[c] = stats.spearmanr(v.iloc[:, 0], v.iloc[:, 1])[0]
    return pd.Series(saida).sort_values(key=np.abs, ascending=False)


def importancia_permutacao(base: BaseDados, detector: DetectorQuebras, serie: str, alvo: str, origem_treino: pd.Timestamp,
                           fim_avaliacao: pd.Timestamp, repeticoes: int = 20, semente: int = 2024) -> pd.DataFrame:
    """Importância por permutação (aumento do MAE) de uma Random Forest, avaliada fora da amostra.

    O modelo é treinado até ``origem_treino`` e avaliado em previsões de 1 passo de
    ``origem_treino + 1`` a ``fim_avaliacao`` (dentro do treino da avaliação principal).
    Grupos de atributos (lags, janelas, calendário, quebras, comportamentais) são permutados juntos.
    """
    from sklearn.ensemble import RandomForestRegressor

    y = base.series[(serie, alvo)]
    c = ConstrutorAtributos(base, detector, serie, origem_treino)
    X_tr, y_tr = c(y.loc[:origem_treino]), y.loc[:origem_treino]
    X_av = c(y.loc[:fim_avaliacao]).loc[origem_treino + pd.Timedelta(days=1):fim_avaliacao, X_tr.columns]
    y_av = y.loc[X_av.index]
    m = RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=5, max_features=0.5, random_state=semente,
                              n_jobs=4).fit(X_tr, y_tr)
    base_mae = np.mean(np.abs(m.predict(X_av) - y_av))
    grupos = {
        "defasagens (lag_k)": [k for k in X_tr if k.startswith("lag_")],
        "janelas móveis (média/DP/CV)": [k for k in X_tr if k.startswith(("media_", "dp_", "cv_"))],
        "calendário e sazonalidade": [k for k in X_tr if k in ("dia_semana", "dia_mes", "mes", "semana_ano", "fim_de_semana",
                                                                "feriado", "data_comercial", "inicio_mes") or k.startswith("fourier")],
        "tendência (tempo)": ["tempo"],
        "dummies de quebra e regimes": [k for k in X_tr if k.startswith(("dummy_", "regime_")) or k == "maquinas_ativas"],
        "comportamentais": [k for k in X_tr if k.startswith("comp_")],
    }
    rng = np.random.default_rng(semente)
    linhas = []
    for grupo, cols in grupos.items():
        if not cols:
            continue
        aumentos = []
        for _ in range(repeticoes):
            Xp = X_av.copy()
            perm = rng.permutation(len(X_tr))[:len(Xp)] if len(X_tr) >= len(Xp) else rng.integers(0, len(X_tr), len(Xp))
            # permuta com valores do treino (quebra a relação com o alvo sem criar valores impossíveis)
            Xp[cols] = X_tr[cols].iloc[perm].to_numpy()
            aumentos.append(np.mean(np.abs(m.predict(Xp) - y_av)) - base_mae)
        linhas.append({"grupo": grupo, "nº de atributos": len(cols), "aumento do MAE (média)": np.mean(aumentos),
                       "aumento do MAE (DP)": np.std(aumentos), "aumento relativo %": 100 * np.mean(aumentos) / base_mae})
    t = pd.DataFrame(linhas).sort_values("aumento do MAE (média)", ascending=False)
    t.attrs["MAE base"] = base_mae
    return t


def contribuicao_dummies(base: BaseDados, detector: DetectorQuebras, serie: str, origem: pd.Timestamp) -> pd.DataFrame:
    """Quanto as dummies obrigatórias explicam da demanda (Poisson/NB2 com dia da semana, treino inteiro).

    Compara, por AIC e razão de verossimilhança, o modelo só com dia da semana contra versões que
    acrescentam ``dummy_horario_reduzido``, ``dummy_novos_clientes`` e as duas (mais regimes intermediários).
    """
    import statsmodels.api as sm

    from src.features.break_dummies import dummies_quebras
    from src.features.temporal_features import dummies_dia_semana

    y = base.series[(serie, "demanda")].loc[:origem]
    D = dummies_quebras(y.index, serie, origem, detector, base.inicio["Máquina 2"])
    X0 = sm.add_constant(dummies_dia_semana(y.index).astype(float))
    variantes = {
        "só dia da semana": [],
        "+ dummy_horario_reduzido": ["dummy_horario_reduzido"],
        "+ dummy_novos_clientes": ["dummy_novos_clientes"],
        "+ as duas dummies obrigatórias": ["dummy_horario_reduzido", "dummy_novos_clientes"],
        "+ dummies + regimes intermediários": [c for c in D if c != "maquinas_ativas"],
    }
    linhas, ll0 = [], None
    for nome, cols in variantes.items():
        cols = [c for c in cols if c in D and D[c].nunique() > 1]
        X = X0.join(D[cols].astype(float)) if cols else X0
        res = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(disp=0, maxiter=500)
        ll0 = res.llf if ll0 is None else ll0
        lr = 2 * (res.llf - ll0)
        from scipy import stats as st
        linhas.append({"modelo": nome, "AIC": res.aic, "LR vs só dia da semana": lr, "gl": len(cols),
                       "p (LR)": st.chi2.sf(lr, len(cols)) if cols else np.nan,
                       "IRR das dummies": ", ".join(f"{c} = {np.exp(res.params[c]):.2f}" for c in cols) or "—",
                       "α (NB2)": res.params.get("alpha", np.nan)})
    t = pd.DataFrame(linhas)
    t["ΔAIC vs só dia da semana"] = t["AIC"] - t["AIC"].iloc[0]
    return t
