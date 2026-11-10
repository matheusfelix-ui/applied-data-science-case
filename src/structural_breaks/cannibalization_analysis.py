"""Quebra inicial: a mudança do horário de funcionamento provocou canibalização?

**Restrição dos dados (decisão documentada).** A janela de vendas da Máquina 1 mudou em
14/05/2024 (de ~10h–20h para ~07h–23h), mas a Máquina 2 só aparece em 08/02/2025. Em 2024
**não existia outra máquina** para ganhar ou perder vendas, então a canibalização *entre máquinas*
causada pela mudança de horário não é testável. A investigação é feita em duas partes separadas:

(a) **Mudança de horário de 2024 — canibalização entre faixas horárias da M1** (porte da seção
    20.1 do notebook principal): as vendas das novas horas são demanda nova ou foram deslocadas
    do núcleo original (10h–19h)? Série temporal interrompida com Binomial Negativa e erros HAC,
    taxa de incrementalidade com bootstrap em blocos e deslocamento de horário por cartão.

(b) **Entrada da Máquina 2 — canibalização entre máquinas**: a M2 tirou vendas da M1? Nível da M1
    antes × depois de 08/02/2025 (controlando a quebra de patamar da M1 de 30/01), por faixa
    horária (a M2 vende mais à noite e no fim de semana), participação de cada máquina e
    correlação diária (substituição gera correlação negativa nos mesmos dias).

Nenhuma outra hipótese operacional é investigada aqui.
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.settings import carregar_config
from src.visualization.plots import CORES, anotar_quebras, cor_serie, formatar_eixo_datas, reservar_topo

_CFG = carregar_config()["quebras"]
FAIXAS = {"Abertura (06–09h)": range(6, 10), "Núcleo manhã (10–13h)": range(10, 14),
          "Núcleo tarde (14–19h)": range(14, 20), "Fechamento (20–23h)": range(20, 24)}
CORES_FAIXAS = dict(zip(FAIXAS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))


def faixa(hora: int) -> str:
    for nome, horas in FAIXAS.items():
        if hora in horas:
            return nome
    return "Fora (00–05h)"


def bootstrap_em_blocos(dados: pd.DataFrame, estatistica, tamanho_bloco: int = 7, n_reamostras: int = 2000,
                        semente: int = _CFG["semente_bootstrap"]) -> np.ndarray:
    """Bootstrap de blocos móveis (preserva a dependência semanal dentro de cada bloco)."""
    rng = np.random.default_rng(semente)
    n = len(dados)
    n_blocos = int(np.ceil(n / tamanho_bloco))
    inicios = np.arange(n - tamanho_bloco + 1)
    saida = []
    for _ in range(n_reamostras):
        idx = np.concatenate([np.arange(i, i + tamanho_bloco) for i in rng.choice(inicios, n_blocos)])[:n]
        saida.append(estatistica(dados.iloc[idx]))
    return np.asarray(saida)


def ajustar_nb(formula: str, dados: pd.DataFrame, maxlags: int = 7):
    """Binomial Negativa (NB2) por máxima verossimilhança com erros-padrão HAC (Newey-West)."""
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        return smf.negativebinomial(formula, dados).fit(disp=0, maxiter=500, cov_type="HAC", cov_kwds={"maxlags": maxlags})


def irr(modelo, termo: str) -> dict:
    """Razão de taxas (IRR) com IC 95% e p-valor."""
    ic = modelo.conf_int().loc[termo]
    return {"IRR": float(np.exp(modelo.params[termo])), "IC95% inf": float(np.exp(ic[0])),
            "IC95% sup": float(np.exp(ic[1])), "p": float(modelo.pvalues[termo])}


# ============================================================ (a) mudança de horário de 2024 (M1)
def base_its(transacoes: pd.DataFrame, corte: pd.Timestamp, semanas: int) -> pd.DataFrame:
    """Demanda diária total, do núcleo (10–19h) e das novas horas, ±``semanas`` em torno do corte (M1)."""
    h0, h1 = _CFG["horas_nucleo_canibalizacao"]
    m1 = transacoes[transacoes["maquina"] == 1].assign(nucleo=lambda d: d["hora"].between(h0, h1))
    inicio, fim = corte - pd.Timedelta(weeks=semanas), corte + pd.Timedelta(weeks=semanas) - pd.Timedelta(days=1)
    cal = pd.date_range(inicio, fim)
    janela = m1[m1["date"].between(inicio, fim)]
    b = janela.groupby("date").agg(total=("money", "size"), nucleo=("nucleo", "sum")).reindex(cal, fill_value=0)
    b["novas_horas"] = b["total"] - b["nucleo"]
    b["pos"] = (b.index >= corte).astype(int)
    b["dia_semana"] = b.index.dayofweek
    b["tempo"] = np.arange(len(b))
    return b


def taxa_incrementalidade(b: pd.DataFrame) -> float:
    """Δ demanda total ÷ demanda das novas horas no pós-período (1 = 100% incremental)."""
    pre, pos = b[b["pos"] == 0], b[b["pos"] == 1]
    return (pos["total"].mean() - pre["total"].mean()) / pos["novas_horas"].mean()


def analisar_mudanca_horario(transacoes: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Quase-experimento da extensão de horário em ±6/8/10 semanas (incrementalidade e IRRs)."""
    linhas = []
    for semanas in _CFG["semanas_its"]:
        b = base_its(transacoes, corte, semanas)
        pre, pos = b[b["pos"] == 0].reset_index(drop=True), b[b["pos"] == 1].reset_index(drop=True)
        m_pre = bootstrap_em_blocos(pre, lambda d: d["total"].mean(), _CFG["tamanho_bloco_bootstrap"], _CFG["n_bootstrap"])
        m_pos = bootstrap_em_blocos(pos, lambda d: (d["total"].mean(), d["novas_horas"].mean()),
                                    _CFG["tamanho_bloco_bootstrap"], _CFG["n_bootstrap"], semente=_CFG["semente_bootstrap"] + 1)
        boot = (m_pos[:, 0] - m_pre) / m_pos[:, 1]
        linha = {"janela": f"±{semanas} semanas", "total pré": pre["total"].mean(), "total pós": pos["total"].mean(),
                 "núcleo pré": pre["nucleo"].mean(), "núcleo pós": pos["nucleo"].mean(),
                 "novas horas pós": pos["novas_horas"].mean(), "incrementalidade": taxa_incrementalidade(b),
                 "IC95% inf": np.percentile(boot, 2.5), "IC95% sup": np.percentile(boot, 97.5)}
        for desfecho in ("total", "nucleo"):
            r = irr(ajustar_nb(f"{desfecho} ~ pos + C(dia_semana)", b), "pos")
            linha[f"IRR {desfecho}"], linha[f"p {desfecho}"] = r["IRR"], r["p"]
            linha[f"IRR {desfecho} IC95%"] = f"[{r['IC95% inf']:.2f}; {r['IC95% sup']:.2f}]"
            r_t = irr(ajustar_nb(f"{desfecho} ~ pos + tempo + C(dia_semana)", b), "pos")
            linha[f"IRR {desfecho} (com tendência)"] = r_t["IRR"]
        linhas.append(linha)
    tabela = pd.DataFrame(linhas).set_index("janela")
    tabela["canibalização estimada"] = 1 - tabela["incrementalidade"]
    return tabela


def deslocamento_cartoes(transacoes: pd.DataFrame, corte: pd.Timestamp, semanas: int = 8) -> dict:
    """Cartões ativos dos dois lados da mudança: migraram de horário? compraram menos no núcleo?"""
    h0, h1 = _CFG["horas_nucleo_canibalizacao"]
    m1 = transacoes[(transacoes["maquina"] == 1) & transacoes["card"].notna()].copy()
    m1 = m1[m1["date"].between(corte - pd.Timedelta(weeks=semanas), corte + pd.Timedelta(weeks=semanas) - pd.Timedelta(days=1))]
    m1["pos"] = m1["date"] >= corte
    m1["novas_horas"] = ~m1["hora"].between(h0, h1)
    ambos = set(m1.loc[~m1["pos"], "card"]) & set(m1.loc[m1["pos"], "card"])
    g = m1[m1["card"].isin(ambos)].groupby(["card", "pos"]).agg(compras=("money", "size"), pct_novas=("novas_horas", "mean")).unstack("pos")
    saida = {"cartões nos dois lados": len(ambos)}
    if len(ambos) >= 10:
        a, d = g[("compras", False)] / semanas, g[("compras", True)] / semanas
        saida.update({"compras/semana pré (mediana)": a.median(), "compras/semana pós (mediana)": d.median(),
                      "Wilcoxon p": stats.wilcoxon(d, a).pvalue,
                      "% compras nas novas horas pré": 100 * g[("pct_novas", False)].mean(),
                      "% compras nas novas horas pós": 100 * g[("pct_novas", True)].mean()})
    novos = m1[m1["pos"] & ~m1["card"].isin(set(m1.loc[~m1["pos"], "card"]))]
    primeiras = novos.sort_values("datetime").groupby("card").head(1)
    saida.update({"cartões novos no pós": primeiras["card"].nunique(),
                  "% 1ªs compras de novos nas novas horas": 100 * primeiras["novas_horas"].mean(),
                  "% de todas as compras pós nas novas horas": 100 * m1.loc[m1["pos"], "novas_horas"].mean()})
    return saida


# ============================================================ (b) entrada da Máquina 2
def analisar_entrada_m2(transacoes: pd.DataFrame, series: dict, entrada_m2: pd.Timestamp,
                        quebra_m1: pd.Timestamp, fim: pd.Timestamp) -> dict:
    """A Máquina 2 tirou vendas da Máquina 1?

    Pré-período = da quebra de patamar da M1 (30/01) à véspera da entrada da M2, para não confundir
    o efeito da M2 com o salto de nível anterior. O pré-período é curto (9 dias): o teste tem baixo
    poder e é complementado por evidências de composição horária e correlação.
    """
    d1 = series[("Máquina 1", "demanda")]
    b = pd.DataFrame({"m1": d1.loc[quebra_m1:fim]})
    b["pos"] = (b.index >= entrada_m2).astype(int)
    b["dia_semana"] = b.index.dayofweek
    nivel = irr(ajustar_nb("m1 ~ pos + C(dia_semana)", b), "pos")

    # Composição horária da M1: regime anterior (desde a quebra de 16/11) × após a entrada da M2
    m1 = transacoes[transacoes["maquina"] == 1].copy()
    m1["faixa"] = m1["hora"].map(faixa)
    inicio_ref = quebra_m1 - pd.Timedelta(weeks=8)
    ref, pos = m1[m1["date"].between(inicio_ref, entrada_m2 - pd.Timedelta(days=1))], m1[m1["date"].between(entrada_m2, fim)]
    dias_ref, dias_pos = (entrada_m2 - inicio_ref).days, (fim - entrada_m2).days + 1
    m2 = transacoes[(transacoes["maquina"] == 2) & transacoes["date"].between(entrada_m2, fim)].copy()
    m2["faixa"] = m2["hora"].map(faixa)
    comp = pd.DataFrame({"M1 antes (trans./dia)": ref["faixa"].value_counts() / dias_ref,
                         "M1 depois (trans./dia)": pos["faixa"].value_counts() / dias_pos,
                         "M2 depois (trans./dia)": m2["faixa"].value_counts() / dias_pos}).reindex(list(FAIXAS)).fillna(0)
    comp["M1 Δ (trans./dia)"] = comp["M1 depois (trans./dia)"] - comp["M1 antes (trans./dia)"]
    comp["% M1 antes"] = 100 * comp["M1 antes (trans./dia)"] / comp["M1 antes (trans./dia)"].sum()
    comp["% M1 depois"] = 100 * comp["M1 depois (trans./dia)"] / comp["M1 depois (trans./dia)"].sum()
    comp["% M2"] = 100 * comp["M2 depois (trans./dia)"] / comp["M2 depois (trans./dia)"].sum()
    tabela_qui = pd.crosstab(pd.concat([ref.assign(p="antes"), pos.assign(p="depois")])["p"],
                             pd.concat([ref, pos])["faixa"])
    qui = stats.chi2_contingency(tabela_qui)

    # Correlação diária M1 × M2 (nula circular) e participação semanal
    d2 = series[("Máquina 2", "demanda")].loc[entrada_m2:fim]
    x, y = d1.loc[entrada_m2:fim].to_numpy(), d2.to_numpy()
    rho = stats.spearmanr(x, y)[0]
    nula = np.array([stats.spearmanr(x, np.roll(y, k))[0] for k in range(1, len(y))])
    # blocos de 7 dias contados a partir da entrada da M2 (evita semana-calendário parcial no início)
    blocos = pd.DataFrame({"M1": d1.loc[entrada_m2:fim], "M2": d2}).resample("7D", origin=entrada_m2)
    sem = blocos.sum()[blocos.size() == 7]          # só blocos completos (o último teria 2 dias)
    sem["% M2"] = 100 * sem["M2"] / (sem["M1"] + sem["M2"])
    tendencia = stats.spearmanr(np.arange(len(sem)), sem["% M2"])
    # Fins de semana: a M2 concentra 41% das vendas no fim de semana; se canibalizasse, o fim de semana da M1 cairia mais
    fds = b.assign(fds=(b["dia_semana"] >= 5).astype(int))
    inter = irr(ajustar_nb("m1 ~ pos * fds", fds), "pos:fds")
    return {"nivel": {"pré-período": f"{quebra_m1:%d/%m}–{entrada_m2 - pd.Timedelta(days=1):%d/%m/%Y}",
                      "média M1 pré": b.loc[b["pos"] == 0, "m1"].mean(), "média M1 pós": b.loc[b["pos"] == 1, "m1"].mean(),
                      **{f"{k} (pós M2)": v for k, v in nivel.items()}},
            "composicao": comp, "qui_quadrado_p": float(qui[1]),
            "correlacao": {"ρ Spearman M1 × M2 (diário)": rho,
                           "p (nula circular)": float((1 + (np.abs(nula) >= abs(rho)).sum()) / len(y))},
            "participacao_semanal": sem, "tendencia_participacao": {"ρ (semana × % M2)": tendencia[0], "p": tendencia[1]},
            "fim_de_semana": {f"{k} (interação pós × fim de semana)": v for k, v in inter.items()}}


def sintese_evidencias(horario: pd.DataFrame, cartoes: dict, m2: dict) -> pd.DataFrame:
    """Evidências a favor e contra a canibalização, nas duas partes da análise."""
    princ = horario.loc[f"±{_CFG['semanas_its_principal']} semanas"]
    comp = m2["composicao"]
    linhas = [
        {"parte": "(a) horário 2024", "evidência": "Núcleo (10–19h) após a extensão",
         "observado": f"{princ['núcleo pré']:.1f} → {princ['núcleo pós']:.1f}/dia (IRR {princ['IRR nucleo']:.2f}, p = {princ['p nucleo']:.2f})",
         "leitura": "contra canibalização forte (núcleo não cai de forma detectável)" if princ["p nucleo"] >= 0.05
                    else "a favor: o núcleo cai após a extensão"},
        {"parte": "(a) horário 2024", "evidência": "Demanda total após a extensão",
         "observado": f"IRR {princ['IRR total']:.2f} (p = {princ['p total']:.3f})", "leitura": "a extensão somou demanda"},
        {"parte": "(a) horário 2024", "evidência": "Taxa de incrementalidade (±6/8/10 sem.)",
         "observado": ", ".join(f"{v:.2f}" for v in horario["incrementalidade"]) +
                      f"; IC95% ±8 sem. [{princ['IC95% inf']:.2f}; {princ['IC95% sup']:.2f}]",
         "leitura": "predominantemente incremental; canibalização parcial não excluída (IC largo)"},
        {"parte": "(a) horário 2024", "evidência": "Cartões recorrentes",
         "observado": f"{cartoes.get('% compras nas novas horas pré', np.nan):.0f}% → {cartoes.get('% compras nas novas horas pós', np.nan):.0f}% das compras nas novas horas; "
                      f"frequência {cartoes.get('compras/semana pré (mediana)', np.nan):.2f} → {cartoes.get('compras/semana pós (mediana)', np.nan):.2f}/sem. (Wilcoxon p = {cartoes.get('Wilcoxon p', np.nan):.2f})",
         "leitura": "há troca de horário individual, sem queda de frequência (n pequeno)"},
        {"parte": "(b) entrada da M2", "evidência": "Nível da M1 após a entrada da M2",
         "observado": f"{m2['nivel']['média M1 pré']:.1f} → {m2['nivel']['média M1 pós']:.1f}/dia (IRR {m2['nivel']['IRR (pós M2)']:.2f}, p = {m2['nivel']['p (pós M2)']:.2f})",
         "leitura": "contra canibalização líquida: sem queda detectável da M1 (pré-período de 9 dias: baixo poder)"
                    if m2["nivel"]["p (pós M2)"] >= 0.05 else "a favor: a M1 cai após a entrada da M2"},
        {"parte": "(b) entrada da M2", "evidência": "M1 no fechamento (20–23h), faixa forte da M2",
         "observado": f"{comp.loc['Fechamento (20–23h)', 'M1 antes (trans./dia)']:.2f} → {comp.loc['Fechamento (20–23h)', 'M1 depois (trans./dia)']:.2f}/dia; "
                      f"composição horária da M1 antes × depois: χ² p = {m2['qui_quadrado_p']:.3f}",
         "leitura": "a favor de deslocamento localizado: a M1 perde no fechamento quase o que a M2 vende nessa faixa; "
                    "confundido com a troca de público da M1 (seção 6.2)"},
        {"parte": "(b) entrada da M2", "evidência": "Fim de semana da M1 (a M2 vende 41% no fim de semana)",
         "observado": f"interação pós × fim de semana: IRR {m2['fim_de_semana']['IRR (interação pós × fim de semana)']:.2f} "
                      f"(p = {m2['fim_de_semana']['p (interação pós × fim de semana)']:.2f})",
         "leitura": ("indício fraco (não significativo a 5%) de queda da M1 nos dias fortes da M2"
                     if m2["fim_de_semana"]["p (interação pós × fim de semana)"] >= 0.05
                     else "a favor: a M1 cai mais nos dias fortes da M2")},
        {"parte": "(b) entrada da M2", "evidência": "Correlação diária M1 × M2",
         "observado": f"ρ = {m2['correlacao']['ρ Spearman M1 × M2 (diário)']:.2f} (p circular = {m2['correlacao']['p (nula circular)']:.2f})",
         "leitura": "substituição diária não detectada (seria ρ negativo)"},
        {"parte": "(b) entrada da M2", "evidência": "Participação da M2 ao longo das semanas",
         "observado": f"ρ = {m2['tendencia_participacao']['ρ (semana × % M2)']:.2f} (p = {m2['tendencia_participacao']['p']:.2f})",
         "leitura": "sem ganho progressivo de participação às custas da M1"},
    ]
    return pd.DataFrame(linhas)


def grafico_mudanca_horario(transacoes: pd.DataFrame, corte: pd.Timestamp) -> plt.Figure:
    """Demanda semanal da M1 em núcleo × novas horas (±8 semanas da extensão)."""
    b = base_its(transacoes, corte, _CFG["semanas_its_principal"])
    sem = b[["nucleo", "novas_horas"]].resample("W-MON", label="left", closed="left").mean()
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.bar(sem.index, sem["nucleo"], width=5, color=cor_serie("Máquina 1"), label="Núcleo original (10–19h)",
           edgecolor="white", lw=1.5)
    ax.bar(sem.index, sem["novas_horas"], width=5, bottom=sem["nucleo"], color="#eda100",
           label="Novas horas (antes das 10h / a partir das 20h)", edgecolor="white", lw=1.5)
    media_pre = b.loc[b["pos"] == 0, "nucleo"].mean()
    # a média entra na legenda (um rótulo sobre a linha ficaria por cima das barras)
    ax.axhline(media_pre, color=CORES["observado"], ls=":", lw=1.1,
               label=f"Média do núcleo no pré-período ({media_pre:.1f}/dia)".replace(".", ","))
    reservar_topo(ax, 0.3)
    anotar_quebras(ax, {f"extensão do horário ({corte:%d/%m/%Y})": corte - pd.Timedelta(days=1)})
    ax.set_ylabel("transações/dia (média da semana)")
    ax.set_title("(a) Máquina 1: as novas horas somaram vendas sem esvaziar o núcleo")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.9))
    formatar_eixo_datas(ax, curto=True)
    fig.tight_layout()
    return fig


def grafico_entrada_m2(m2: dict, series: dict, entrada_m2: pd.Timestamp, quebra_m1: pd.Timestamp) -> plt.Figure:
    """(b) Demanda das máquinas e total em torno da entrada da M2, composição horária e participação."""
    fig, eixos = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1.6, 1.1, 1]})
    ax = eixos[0]
    inicio = quebra_m1 - pd.Timedelta(weeks=8)
    for serie in ("Máquina 1", "Máquina 2", "Total"):
        s = series[(serie, "demanda")].loc[inicio:]
        ax.plot(s.index, s.rolling(7, min_periods=4).mean(), color=cor_serie(serie), lw=2.2, label=f"{serie} (média 7 dias)")
    anotar_quebras(ax, {"quebra M1": quebra_m1, "entrada M2": entrada_m2})
    ax.set_ylabel("transações/dia")
    ax.set_title("(b) Demanda em torno da entrada da M2")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.88), fontsize=8)
    formatar_eixo_datas(ax, curto=True)
    ax = eixos[1]
    comp = m2["composicao"]
    x = np.arange(len(comp))
    for k, (col, cor, rot) in enumerate([("M1 antes (trans./dia)", CORES["contexto"], "M1 antes"),
                                         ("M1 depois (trans./dia)", cor_serie("Máquina 1"), "M1 depois"),
                                         ("M2 depois (trans./dia)", cor_serie("Máquina 2"), "M2 depois")]):
        ax.bar(x + (k - 1) * 0.27, comp[col], width=0.25, color=cor, label=rot)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(" (")[0] + "\n(" + n.split(" (")[1] for n in comp.index], fontsize=8)
    ax.set_ylabel("transações/dia")
    ax.set_title("Vendas por faixa horária")
    ax.legend(loc="upper left", fontsize=8)
    ax = eixos[2]
    sem = m2["participacao_semanal"]
    ax.bar(sem.index, sem["% M2"], width=5, color=cor_serie("Máquina 2"))
    ax.set_ylabel("% da demanda total no bloco")
    ax.set_title("Participação da M2 (blocos de 7 dias)")
    formatar_eixo_datas(ax, curto=True)
    fig.tight_layout()
    return fig
