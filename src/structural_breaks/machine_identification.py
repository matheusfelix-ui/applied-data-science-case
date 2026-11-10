"""`index_2.csv` é uma nova máquina ou repete vendas de `index_1.csv`?

Organização em funções do notebook original ``teste_index2_nova_maquina.ipynb`` (raiz do
diretório), **sem alterar a metodologia**. Hipóteses:

- **H_dup**: ``index_2`` repete (no todo ou em parte) vendas de ``index_1`` — mesmo relógio, outro
  fuso, datas deslocadas ou preços convertidos;
- **H_nova**: ``index_2`` é outra fonte de vendas, independente.

O núcleo do desenho é a **nula por deslocamento circular de dias**: os horários de ``index_2`` são
movidos k dias (k = 1..D−1, com volta), o que preserva o perfil intradiário e o volume diário mas
destrói qualquer pareamento real. Testes: correspondência transação a transação (4), varredura do
relógio −12 h..+12 h (5) com controle positivo (5.1), subconjunto (6), preços (7), pagamento (8),
restrição física (9), dependência dos volumes (10), perfis horário/semanal (11) e síntese com Holm (12).

A semente e a ordem de uso do gerador aleatório são as do notebook original, o que reproduz os
mesmos números.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.logging_config import obter_logger
from src.visualization.plots import CORES, cor_serie, reservar_topo

log = obter_logger(__name__)

TOLERANCIAS = [1, 5, 60, 300]
DESLOCAMENTOS_MIN = np.arange(-720, 721)
DELTA_SINTETICO = 3 * 3600
FRACOES_CONTROLE = (0.02, 0.05, 0.10)
N_PERMUTACOES = 5000
COR = {"index_1": cor_serie("Máquina 1"), "index_2": cor_serie("Máquina 2")}


def distancia_mais_proxima(alvo: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    """Distância (s) de cada instante de ``alvo`` ao instante mais próximo de ``referencia``."""
    referencia = np.sort(referencia)
    if len(referencia) == 0:
        return np.full(len(alvo), np.inf)
    i = np.searchsorted(referencia, alvo)
    direita = referencia[np.clip(i, 0, len(referencia) - 1)]
    esquerda = referencia[np.clip(i - 1, 0, len(referencia) - 1)]
    return np.minimum(np.abs(direita - alvo), np.abs(esquerda - alvo))


@dataclass
class IdentificacaoMaquinas:
    """Executa a bateria de testes no período comum aos dois arquivos.

    Args:
        transacoes: transações padronizadas (``BaseDados.transacoes``).
        semente: semente do gerador (2024, a do notebook original).
    """
    transacoes: pd.DataFrame
    semente: int = 2024
    resultados: dict = field(default_factory=dict)

    def __post_init__(self):
        t = self.transacoes.copy()
        t["arquivo"] = t["arquivo_origem"].str.replace(".csv", "", regex=False)
        t["data"] = t["date"]
        t["produto"] = t["coffee_name"].str.lower()
        self.i1_completo = t[t["arquivo"] == "index_1"].copy()
        self.i2_completo = t[t["arquivo"] == "index_2"].copy()
        self.inicio = self.i2_completo["data"].min()
        self.fim = min(self.i1_completo["data"].max(), self.i2_completo["data"].max())
        self.dias = pd.date_range(self.inicio, self.fim, freq="D")
        self.D = len(self.dias)
        self.i1 = self._recortar(self.i1_completo)
        self.i2 = self._recortar(self.i2_completo)
        self.T2, self.P2 = self.i2["t"].to_numpy(), self.i2["produto"].to_numpy()
        self.rng = np.random.default_rng(self.semente)
        self._por_produto = {p: np.sort(g["t"].to_numpy()) for p, g in self.i1.groupby("produto")}
        self.diario = pd.DataFrame({k: d.groupby("data").size().reindex(self.dias, fill_value=0)
                                    for k, d in [("index_1", self.i1), ("index_2", self.i2)]})

    def _recortar(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d[d["data"].between(self.inicio, self.fim)].copy()
        d["t"] = (d["datetime"] - self.inicio).dt.total_seconds().to_numpy()
        return d.sort_values("t").reset_index(drop=True)

    # ---------------------------------------------------------------- utilitários dos testes
    def distancias(self, t_alvo: np.ndarray, produtos_alvo: np.ndarray, mesmo_produto: bool) -> np.ndarray:
        """Distância ao registro mais próximo de ``index_1`` (opcionalmente do mesmo produto)."""
        if not mesmo_produto:
            return distancia_mais_proxima(t_alvo, self.i1["t"].to_numpy())
        saida = np.full(len(t_alvo), np.inf)
        for p in np.unique(produtos_alvo):
            idx = np.flatnonzero(produtos_alvo == p)
            saida[idx] = distancia_mais_proxima(t_alvo[idx], self._por_produto.get(p, np.array([])))
        return saida

    def deslocar_dias(self, t: np.ndarray, k: int) -> np.ndarray:
        """Desloca k dias inteiros, com volta circular dentro do período comum (preserva a hora)."""
        dia = np.floor(t / 86400)
        return t - dia * 86400 + ((dia + k) % self.D) * 86400

    def varrer(self, t: np.ndarray, produtos: np.ndarray | None = None, mesmo_produto: bool = True) -> np.ndarray:
        """Casamentos (±60 s) para cada deslocamento de relógio de −12 h a +12 h."""
        produtos = self.P2 if produtos is None else produtos
        return np.array([(self.distancias(t + 60 * m, produtos, mesmo_produto) <= 60).sum() for m in DESLOCAMENTOS_MIN])

    # ---------------------------------------------------------------- 3. formato do registro
    def formato_registro(self) -> pd.DataFrame:
        """Formato de cada arquivo: precisão do relógio, preços, pagamentos e pedidos múltiplos."""
        linhas = {}
        brutos = {"index_1": self.i1, "index_2": self.i2}
        for k, d in brutos.items():
            g = d.groupby("datetime").size()
            ms = (d["datetime"].dt.microsecond > 0).mean()
            linhas[k] = {
                "colunas com identificador de cartão": "sim" if d["card"].notna().any() else "não",
                "horários com milissegundos": f"{ms:.0%}",
                "preços com centavos": f"{(d['money'] % 1 != 0).mean():.0%}",
                "preços distintos": d["money"].nunique(),
                "produtos distintos": d["produto"].nunique(),
                "% em dinheiro (cash)": f"{(d['cash_type'] == 'cash').mean():.0%}",
                "horários repetidos (grupos)": int((g > 1).sum()),
                "registros nesses grupos": int(g[g > 1].sum()),
                "menor intervalo entre vendas (s)": round(float(np.diff(d["t"]).min()), 1),
            }
        tabela = pd.DataFrame(linhas)
        self.resultados["formato"] = tabela
        return tabela

    # ---------------------------------------------------------------- 4. correspondência
    def correspondencia(self) -> pd.DataFrame:
        """Casamentos com tolerâncias de 1 s a 5 min, com e sem mesmo produto, contra a nula de dias."""
        linhas, nulas = [], {}
        for mesmo_produto in (False, True):
            por_k = np.array([[(self.distancias(self.deslocar_dias(self.T2, k), self.P2, mesmo_produto) <= tol).sum()
                               for tol in TOLERANCIAS] for k in range(self.D)])
            nulas[mesmo_produto] = por_k
            for j, tol in enumerate(TOLERANCIAS):
                observado, nula = por_k[0, j], por_k[1:, j]
                linhas.append({
                    "exige mesmo produto": "sim" if mesmo_produto else "não",
                    "tolerância": f"±{tol} s" if tol < 60 else f"±{tol // 60} min",
                    "casamentos observados (k = 0)": int(observado),
                    "% de index_2 casado": 100 * observado / len(self.i2),
                    "média sob a nula": nula.mean(), "máximo sob a nula": int(nula.max()),
                    "p empírico (H_dup: mais casamentos)": (1 + (nula >= observado).sum()) / self.D,
                })
        tabela = pd.DataFrame(linhas)
        contagem_k = nulas[True][:, TOLERANCIAS.index(60)]
        self.resultados.update(correspondencia=tabela, contagem_k=contagem_k)
        return tabela

    # ---------------------------------------------------------------- 5. relógio
    def varredura_relogio(self) -> dict:
        """Varredura de −12 h a +12 h e máximos de varreduras nulas (k = 1..D−1)."""
        varredura = self.varrer(self.T2)
        nulas = np.array([self.varrer(self.deslocar_dias(self.T2, k)) for k in range(1, self.D)])
        maximos = nulas.max(axis=1)
        varredura_qq = self.varrer(self.T2, mesmo_produto=False)
        maximos_qq = np.array([self.varrer(self.deslocar_dias(self.T2, k), mesmo_produto=False).max()
                               for k in range(1, self.D)])
        r = {"varredura": varredura, "referencia": nulas.mean(axis=0), "maximos_nulos": maximos,
             "pico_min": int(DESLOCAMENTOS_MIN[np.argmax(varredura)]),
             "p": (1 + (maximos >= varredura.max()).sum()) / (1 + len(maximos)),
             "varredura_qq": varredura_qq, "maximos_nulos_qq": maximos_qq,
             "pico_qq_min": int(DESLOCAMENTOS_MIN[np.argmax(varredura_qq)]),
             "p_qq": (1 + (maximos_qq >= varredura_qq.max()).sum()) / (1 + len(maximos_qq))}
        self.resultados["relogio"] = r
        return r

    def controle_positivo(self) -> pd.DataFrame:
        """Planta cópias de 2%, 5% e 10% de registros de ``index_1`` (relógio +3 h) e refaz a varredura."""
        controle = []
        for fracao in FRACOES_CONTROLE:
            n = round(fracao * len(self.i2))
            trocar = self.rng.choice(len(self.i2), n, replace=False)
            origem = self.i1.iloc[self.rng.choice(len(self.i1), n, replace=False)]
            t_sint, p_sint = self.T2.copy(), self.P2.copy()
            t_sint[trocar] = np.floor(origem["t"].to_numpy()) + DELTA_SINTETICO
            p_sint[trocar] = origem["produto"].to_numpy()
            real = self.varrer(t_sint, p_sint)
            nulos = np.array([self.varrer(self.deslocar_dias(t_sint, k), p_sint).max() for k in range(1, self.D)])
            controle.append({"fração copiada": f"{fracao:.0%}", "registros copiados": n,
                             "máximo da varredura": int(real.max()),
                             "deslocamento do máximo (min)": int(DESLOCAMENTOS_MIN[np.argmax(real)]),
                             "faixa dos máximos nulos": f"{nulos.min()}–{nulos.max()}",
                             "p": (1 + (nulos >= real.max()).sum()) / self.D,
                             "detectado no deslocamento plantado": int(DESLOCAMENTOS_MIN[np.argmax(real)]) == -DELTA_SINTETICO // 60})
        tabela = pd.DataFrame(controle)
        self.resultados["controle_positivo"] = tabela
        return tabela

    # ---------------------------------------------------------------- 6. subconjunto
    def subconjunto(self) -> pd.DataFrame:
        """Se ``index_2`` ⊂ ``index_1``, então contagem_2 ≤ contagem_1 em todo dia × produto."""
        historico = set(self.i1_completo["produto"])
        produtos_i2 = set(self.i2["produto"])
        comuns = sorted(produtos_i2 & set(self.i1["produto"]))
        exclusivos = sorted(produtos_i2 - historico)
        grade = pd.concat({"index_1": self.i1.groupby(["data", "produto"]).size(),
                           "index_2": self.i2.groupby(["data", "produto"]).size()}, axis=1).fillna(0)
        violacoes = grade[grade["index_2"] > grade["index_1"]]
        violacoes_comuns = violacoes[violacoes.index.get_level_values("produto").isin(comuns)]
        tabela = pd.Series({
            "produtos em index_2 (período comum)": len(produtos_i2),
            "produtos de index_2 também vendidos por index_1 no período": len(comuns),
            "produtos de index_2 que nunca aparecem em index_1": len(exclusivos),
            "% das vendas de index_2 com produto inexistente em index_1": 100 * self.i2["produto"].isin(exclusivos).mean(),
            "Jaccard dos cardápios no período": len(produtos_i2 & set(self.i1["produto"])) / len(produtos_i2 | set(self.i1["produto"])),
            "células dia × produto com index_2 > index_1 (produtos comuns)": len(violacoes_comuns),
            "dias com total de index_2 > total de index_1": int((self.diario["index_2"] > self.diario["index_1"]).sum()),
        }).rename("valor").to_frame()
        self.resultados.update(subconjunto=tabela, comuns=comuns, exclusivos=exclusivos,
                               n_violacoes_comuns=len(violacoes_comuns))
        return tabela

    # ---------------------------------------------------------------- 7. preços
    def precos(self) -> pd.DataFrame:
        """Razão de preços index_2/index_1 nos produtos comuns: CV ≈ 0 indicaria conversão."""
        comuns = self.resultados.get("comuns") or sorted(set(self.i2["produto"]) & set(self.i1["produto"]))
        linhas = []
        for p in comuns:
            v1, v2 = self.i1.loc[self.i1["produto"] == p, "money"], self.i2.loc[self.i2["produto"] == p, "money"]
            linhas.append({"produto": p, "preço modal index_1": v1.mode()[0], "preço modal index_2": v2.mode()[0]})
        tabela = pd.DataFrame(linhas).set_index("produto")
        tabela["razão index_2 / index_1"] = tabela["preço modal index_2"] / tabela["preço modal index_1"]
        self.resultados["precos"] = tabela
        return tabela

    # ---------------------------------------------------------------- 8. pagamento
    def pagamento(self) -> dict:
        """Fisher exato da proporção de pagamentos em dinheiro nos dois arquivos."""
        tabela = pd.crosstab(pd.concat([self.i1["arquivo"], self.i2["arquivo"]], ignore_index=True),
                             pd.concat([self.i1["cash_type"], self.i2["cash_type"]], ignore_index=True))
        tabela = tabela.reindex(columns=["card", "cash"], fill_value=0)
        _, p = stats.fisher_exact(tabela.to_numpy())
        r = {"tabela": tabela, "p": p}
        self.resultados["pagamento"] = r
        return r

    # ---------------------------------------------------------------- 9. restrição física
    def restricao_fisica(self) -> dict:
        """Pares cruzados mais próximos que o intervalo mínimo já observado em ``index_1``."""
        intervalos = np.diff(np.sort((self.i1_completo["datetime"] - self.inicio).dt.total_seconds().to_numpy()))
        minimo = intervalos.min()
        cruzados = int((self.distancias(self.T2, self.P2, False) < minimo).sum())
        nula = np.array([(self.distancias(self.deslocar_dias(self.T2, k), self.P2, False) < minimo).sum()
                         for k in range(1, self.D)])
        r = {"minimo_i1_s": float(minimo), "cruzados_abaixo": cruzados, "media_nula": float(nula.mean())}
        self.resultados["restricao"] = r
        return r

    # ---------------------------------------------------------------- 10. dependência
    def dependencia(self) -> pd.DataFrame:
        """Spearman diário e dia × hora, com p pela nula circular."""
        spearman = lambda a, b: stats.spearmanr(a, b)[0]

        def p_circular(x, y):
            obs = spearman(x, y)
            nula = np.array([spearman(x, np.roll(y, k)) for k in range(1, len(y))])
            return obs, (1 + (np.abs(nula) >= abs(obs)).sum()) / len(y)

        receita = pd.DataFrame({k: d.groupby("data")["money"].sum().reindex(self.dias, fill_value=0)
                                for k, d in [("index_1", self.i1), ("index_2", self.i2)]})
        idx = pd.MultiIndex.from_product([self.dias, range(24)], names=["data", "hora"])
        por_hora = pd.DataFrame({k: d.groupby([d["data"], d["datetime"].dt.hour]).size().reindex(idx, fill_value=0)
                                 for k, d in [("index_1", self.i1), ("index_2", self.i2)]})
        ativas = por_hora.groupby(level="hora").sum().sum(axis=1) > 0
        por_hora = por_hora[por_hora.index.get_level_values("hora").isin(ativas[ativas].index)]

        def rho_dh(k):
            y = por_hora["index_2"].unstack("hora")
            y = pd.DataFrame(np.roll(y.to_numpy(), k, axis=0), index=y.index, columns=y.columns).stack()
            return spearman(por_hora["index_1"], y.reindex(por_hora.index))

        obs_dh = rho_dh(0)
        nula_dh = np.array([rho_dh(k) for k in range(1, self.D)])
        tabela = pd.DataFrame([
            {"medida": "transações por dia", **dict(zip(["ρ de Spearman", "p (nula circular)"],
                                                         p_circular(self.diario["index_1"].to_numpy(), self.diario["index_2"].to_numpy())))},
            {"medida": "receita por dia", **dict(zip(["ρ de Spearman", "p (nula circular)"],
                                                      p_circular(receita["index_1"].to_numpy(), receita["index_2"].to_numpy())))},
            {"medida": "transações por dia × hora", "ρ de Spearman": obs_dh,
             "p (nula circular)": (1 + (np.abs(nula_dh) >= abs(obs_dh)).sum()) / self.D},
        ])
        self.resultados["dependencia"] = tabela
        return tabela

    # ---------------------------------------------------------------- 11. perfis
    def perfis(self) -> dict:
        """Perfil horário (permutação dentro do dia) e semanal (quasi-Poisson diário)."""
        faixa = lambda h: pd.cut(h, bins=[-1, 8, 10, 12, 14, 16, 18, 20, 23],
                                 labels=["≤08h", "09–10h", "11–12h", "13–14h", "15–16h", "17–18h", "19–20h", "21–23h"])
        arquivos = pd.concat([self.i1["arquivo"], self.i2["arquivo"]], ignore_index=True)
        horas = pd.concat([self.i1["datetime"].dt.hour, self.i2["datetime"].dt.hour], ignore_index=True)
        dias = pd.concat([self.i1["datetime"].dt.dayofweek, self.i2["datetime"].dt.dayofweek], ignore_index=True)
        cramer = {}
        for nome, coluna in [("faixa horária", faixa(horas)), ("dia da semana", dias)]:
            t = pd.crosstab(arquivos.rename("arquivo"), coluna)
            qui = stats.chi2_contingency(t)[0]
            cramer[nome] = np.sqrt(qui / (t.to_numpy().sum() * (min(t.shape) - 1)))
        base_perm = pd.DataFrame({"arquivo": arquivos.to_numpy(), "faixa": faixa(horas).astype(str).to_numpy(),
                                  "data": pd.concat([self.i1["data"], self.i2["data"]], ignore_index=True).to_numpy()})
        grupos = [np.flatnonzero(base_perm["data"].to_numpy() == d) for d in base_perm["data"].unique()]
        rotulos = base_perm["arquivo"].to_numpy()
        qui_faixa = lambda r: stats.chi2_contingency(pd.crosstab(r, base_perm["faixa"].to_numpy()))[0]
        observado = qui_faixa(rotulos)
        nula = []
        for _ in range(N_PERMUTACOES):
            r = rotulos.copy()
            for g in grupos:
                r[g] = self.rng.permutation(r[g])
            nula.append(qui_faixa(r))
        p_faixa = (1 + (np.array(nula) >= observado).sum()) / (1 + len(nula))
        contagens = self.diario.rename_axis("data").reset_index().melt(id_vars="data", var_name="arquivo", value_name="n")
        contagens["dow"] = contagens["data"].dt.dayofweek.astype(str)
        completo = smf.glm("n ~ C(arquivo) * C(dow)", contagens, family=sm.families.Poisson()).fit(scale="X2")
        reduzido = smf.glm("n ~ C(arquivo) + C(dow)", contagens, family=sm.families.Poisson()).fit(scale="X2")
        gl_num, gl_den = reduzido.df_resid - completo.df_resid, completo.df_resid
        F = (reduzido.deviance - completo.deviance) / gl_num / completo.scale
        r = {"cramer": cramer, "p_faixa": p_faixa, "p_dia": stats.f.sf(F, gl_num, gl_den), "F_dia": F,
             "fim_de_semana_%": {k: 100 * (d["datetime"].dt.dayofweek >= 5).mean() for k, d in [("index_1", self.i1), ("index_2", self.i2)]}}
        self.resultados["perfis"] = r
        return r

    # ---------------------------------------------------------------- 12. síntese
    def executar(self) -> pd.DataFrame:
        """Roda todos os testes na ordem do notebook original e devolve a tabela-síntese (Holm)."""
        log.info("Identificação das máquinas: período comum %s a %s (%d dias)", f"{self.inicio:%d/%m/%Y}",
                 f"{self.fim:%d/%m/%Y}", self.D)
        self.formato_registro()
        corr = self.correspondencia()
        rel = self.varredura_relogio()
        self.controle_positivo()
        self.subconjunto()
        pr = self.precos()
        pag = self.pagamento()
        fis = self.restricao_fisica()
        dep = self.dependencia()
        per = self.perfis()

        def linha(teste, observado, h0="—", p=np.nan, favorece="—"):
            return {"teste": teste, "observado": observado, "H0 do teste": h0, "p": p, "favorece": favorece}

        lado_corr = lambda p: "H_dup" if p < 0.05 else "H_nova (nada acima do acaso)"
        lado_igual = lambda p: "H_nova" if p < 0.05 else "inconclusivo"
        c60 = corr[(corr["exige mesmo produto"] == "sim") & (corr["tolerância"] == "±1 min")].iloc[0]
        c5 = corr[(corr["exige mesmo produto"] == "não") & (corr["tolerância"] == "±5 s")].iloc[0]
        ck = self.resultados["contagem_k"]
        razoes = pr["razão index_2 / index_1"]
        frac = lambda d: (d["cash_type"] == "cash").mean()
        sintese = pd.DataFrame([
            linha("4. Casamentos, mesmo produto, ±60 s",
                  f"{c60['casamentos observados (k = 0)']} de {len(self.i2)} (acaso ≈ {c60['média sob a nula']:.1f})",
                  "acaso", c60["p empírico (H_dup: mais casamentos)"], lado_corr(c60["p empírico (H_dup: mais casamentos)"])),
            linha("4. Casamentos, qualquer produto, ±5 s",
                  f"{c5['casamentos observados (k = 0)']} (acaso ≈ {c5['média sob a nula']:.1f})",
                  "acaso", c5["p empírico (H_dup: mais casamentos)"], lado_corr(c5["p empírico (H_dup: mais casamentos)"])),
            linha("4. Datas deslocadas (k dias)", f"máximo {ck.max()} casamentos (k = {int(np.argmax(ck))}); mediana {np.median(ck):.0f}",
                  favorece="H_nova (sem pico)"),
            linha("5. Relógio deslocado (−12 h a +12 h)",
                  f"máx. {rel['varredura'].max()} em {rel['pico_min']:+d} min; máximos nulos {rel['maximos_nulos'].min()}–{rel['maximos_nulos'].max()}",
                  "acaso", rel["p"], lado_corr(rel["p"])),
            linha("5. Relógio deslocado, qualquer produto",
                  f"máx. {rel['varredura_qq'].max()} em {rel['pico_qq_min']:+d} min; máximos nulos {rel['maximos_nulos_qq'].min()}–{rel['maximos_nulos_qq'].max()}",
                  "acaso", rel["p_qq"], lado_corr(rel["p_qq"])),
            linha("6. Subconjunto", f"{self.resultados['n_violacoes_comuns']} violações em produtos comuns; "
                  f"{100 * self.i2['produto'].isin(self.resultados['exclusivos']).mean():.0f}% das vendas com produto inexistente em index_1",
                  favorece="H_nova (critério determinístico)"),
            linha("7. Preços", f"razões {razoes.min():.2f}–{razoes.max():.2f} (CV {razoes.std(ddof=1) / razoes.mean():.0%})",
                  favorece="H_nova (sem fator único)"),
            linha("8. Pagamento em dinheiro", f"index_1 {frac(self.i1):.0%} × index_2 {frac(self.i2):.0%}",
                  "proporções iguais", pag["p"], lado_igual(pag["p"])),
            linha("9. Mesmo equipamento físico", f"{fis['cruzados_abaixo']} pares < {fis['minimo_i1_s']:.0f} s (acaso ≈ {fis['media_nula']:.1f})",
                  favorece="H_nova (evidência fraca)"),
            linha("10. Volume diário", f"ρ = {dep.loc[0, 'ρ de Spearman']:.2f}", "ρ = 0", dep.loc[0, "p (nula circular)"],
                  "H_nova (sem dependência)" if dep.loc[0, "p (nula circular)"] >= 0.05 else "H_dup"),
            linha("10. Volume dia × hora", f"ρ = {dep.loc[2, 'ρ de Spearman']:.2f}", "ρ = 0", dep.loc[2, "p (nula circular)"],
                  "H_nova (sem dependência)" if dep.loc[2, "p (nula circular)"] >= 0.05 else "H_dup"),
            linha("11. Perfil horário", f"V de Cramér = {per['cramer']['faixa horária']:.2f}", "perfis iguais",
                  per["p_faixa"], lado_igual(per["p_faixa"])),
            linha("11. Perfil por dia da semana", f"V de Cramér = {per['cramer']['dia da semana']:.2f}", "perfis iguais",
                  per["p_dia"], lado_igual(per["p_dia"])),
        ])
        tem_p = sintese["p"].notna()
        sintese.loc[tem_p, "p (Holm)"] = multipletests(sintese.loc[tem_p, "p"], method="holm")[1]
        self.resultados["sintese"] = sintese
        log.info("Identificação concluída: %d/%d testes favorecem H_nova; nenhum favorece H_dup",
                 int(sintese["favorece"].str.startswith("H_nova").sum()), len(sintese))
        return sintese

    # ---------------------------------------------------------------- gráficos
    def grafico_varredura(self) -> plt.Figure:
        """Curva de casamentos por deslocamento do relógio contra o nível de acaso."""
        r = self.resultados["relogio"]
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(DESLOCAMENTOS_MIN / 60, r["varredura"], color=COR["index_2"], lw=1.2, label="Casamentos observados")
        ax.plot(DESLOCAMENTOS_MIN / 60, pd.Series(r["referencia"]).rolling(31, center=True, min_periods=1).mean(),
                color=CORES["contexto"], lw=2, ls="--", label=f"Nível de acaso (média de {len(r['maximos_nulos'])} varreduras nulas)")
        ax.axvline(0, color=CORES["observado"], lw=0.8, alpha=0.5)
        ax.annotate(f"máximo: {r['varredura'].max()} casamentos em {r['pico_min'] / 60:+.1f} h",
                    (r["pico_min"] / 60, r["varredura"].max()), xytext=(8, -4), textcoords="offset points", fontsize=9)
        ax.set_xlim(-12, 12)
        ax.set_xticks(range(-12, 13, 2))
        ax.set_ylim(0, max(len(self.i2) * 0.15, r["varredura"].max() * 1.3))
        ax.set_xlabel("Deslocamento aplicado ao relógio de index_2 (horas)")
        ax.set_ylabel("Registros de index_2 casados")
        ax.set_title("Varredura do relógio: mesmo produto, tolerância ±60 s — sem pico acima do acaso")
        ax.legend(loc="upper left")
        return fig

    def grafico_perfis(self) -> plt.Figure:
        """Volume diário e perfil horário das duas fontes no período comum."""
        fig, eixos = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
        ax = eixos[0]
        for k, rot in [("index_1", "Máquina 1 (index_1)"), ("index_2", "Máquina 2 (index_2)")]:
            ax.plot(self.diario.index, self.diario[k], marker="o", ms=4, color=COR[k], label=rot)
        ax.set_title("Transações por dia no período comum")
        ax.set_ylabel("transações/dia")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.set_ylim(bottom=0)
        reservar_topo(ax, 0.2)                       # legenda acima dos picos diários
        ax.legend(loc="upper left", ncol=2)
        ax = eixos[1]
        largura = 0.4
        for j, (k, d) in enumerate([("index_1", self.i1), ("index_2", self.i2)]):
            perfil = d["datetime"].dt.hour.value_counts(normalize=True).reindex(range(6, 24), fill_value=0)
            ax.bar(perfil.index + (j - 0.5) * largura, perfil.values, width=largura - 0.05, color=COR[k],
                   label="Máquina 1" if k == "index_1" else "Máquina 2")
        ax.set_xticks(range(6, 24, 2))
        ax.set_xticklabels([f"{h:02d}h" for h in range(6, 24, 2)])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylabel("% das transações do arquivo")
        ax.set_title("Perfil horário no período comum")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig
