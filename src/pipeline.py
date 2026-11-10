"""Pipeline ponta a ponta do projeto Coffee Sales.

Uso:
    python -m src.pipeline                       # todas as etapas
    python -m src.pipeline --etapas dados eda    # etapas escolhidas
    python -m src.pipeline --rapido              # sem backtest (CI / verificação rápida)

Etapas (cada uma grava tabelas CSV e gráficos PNG/PDF em ``reports/``):

1. ``dados``          carga, validação, padronização e séries diárias → ``data/processed``
2. ``identificacao``  index_2 é outra máquina? (testes reaproveitados) → ``reports/structural_breaks``
3. ``eda``            descritiva, ausentes, zeros, outliers, produtos → ``reports/eda``
4. ``quebras``        testes de quebra, novos clientes, canibalização → ``reports/structural_breaks``
5. ``features``       atributos, testes de vazamento, contribuição das dummies → ``reports/feature_analysis``
6. ``modelagem``      6 modelos × 6 séries na origem principal (teste 10–23/03/2025)
7. ``backtest``       11 origens de 14 dias (M1 e Total)
8. ``avaliacao``      métricas, significância, resíduos, treino×validação×teste, seleção → ``reports/model_comparison``
9. ``previsoes``      previsões finais (teste + 14 dias futuros) e modelos salvos → ``reports/forecasts``, ``models/``

Os resultados intermediários ficam em ``data/processed/artefatos/*.pkl`` para que os notebooks
não precisem recalcular as etapas pesadas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessing import BaseDados, preparar_base
from src.logging_config import obter_logger
from src.settings import CONFIG_PADRAO, RAIZ_PROJETO, caminho, carregar_config, pasta_relatorio
from src.visualization.plots import cor_serie, plot_previsao, salvar_figura

log = obter_logger("src.pipeline")
CFG = carregar_config()
SERIES = [(s, a) for s in CFG["series"]["nomes"] for a in CFG["series"]["alvos"]]
ETAPAS = ["dados", "identificacao", "eda", "quebras", "features", "modelagem", "backtest", "avaliacao", "previsoes"]


# ============================================================================ utilitários
def pasta_artefatos() -> Path:
    p = caminho("dados_processados") / "artefatos"
    p.mkdir(parents=True, exist_ok=True)
    return p


def salvar_artefato(nome: str, obj) -> None:
    with open(pasta_artefatos() / f"{nome}.pkl", "wb") as f:
        pickle.dump(obj, f)


def carregar_artefato(nome: str):
    arquivo = pasta_artefatos() / f"{nome}.pkl"
    if not arquivo.exists():
        raise FileNotFoundError(f"Artefato '{nome}' não encontrado: rode a etapa correspondente do pipeline")
    with open(arquivo, "rb") as f:
        return pickle.load(f)


def salvar_tabela(df: pd.DataFrame, pasta: Path, nome: str, indice: bool = False) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    df.to_csv(pasta / f"{nome}.csv", index=indice, float_format="%.6g")


def hash_config() -> str:
    return hashlib.sha256(CONFIG_PADRAO.read_bytes()).hexdigest()[:12]


def _datas(base: BaseDados) -> dict:
    from src.validation.backtesting import datas_divisao

    return datas_divisao(base)


# ============================================================================ etapas
def etapa_dados() -> BaseDados:
    base = preparar_base(salvar=True)
    salvar_artefato("base", base)
    return base


def etapa_identificacao(base: BaseDados) -> None:
    from src.structural_breaks.machine_identification import IdentificacaoMaquinas

    pasta = pasta_relatorio("structural_breaks", "identificacao_maquinas")
    idm = IdentificacaoMaquinas(base.transacoes)
    sintese = idm.executar()
    salvar_tabela(sintese, pasta, "sintese_testes")
    for nome in ("formato", "correspondencia", "controle_positivo", "subconjunto", "precos", "dependencia"):
        salvar_tabela(idm.resultados[nome], pasta, nome, indice=nome in ("formato", "subconjunto", "precos"))
    salvar_figura(idm.grafico_varredura(), pasta, "varredura_relogio")
    salvar_figura(idm.grafico_perfis(), pasta, "perfis_periodo_comum")
    salvar_artefato("identificacao", {"sintese": sintese, **{k: v for k, v in idm.resultados.items() if k != "relogio"},
                                      "relogio": {k: v for k, v in idm.resultados["relogio"].items()}})


def etapa_eda(base: BaseDados) -> None:
    from src.eda import descriptive_analysis as da, missing_values as mv, outlier_analysis as oa, product_analysis as pa, zero_analysis as za
    from src.eda import duplicate_analysis as du, trend_seasonality as ts
    from src.structural_breaks.structural_break_test import detectar_quebras_nivel

    pasta = pasta_relatorio("eda")
    origem = _datas(base)["origem"]
    quebras = {s: detectar_quebras_nivel(base.series[(s, "demanda")].loc[:origem])[0] for s in ("Máquina 1", "Total")}
    rot = {s: {f"quebra {q:%d/%m/%y}": q for q in qs} for s, qs in quebras.items()}
    salvar_tabela(da.resumo_series(base), pasta, "resumo_series")
    salvar_tabela(da.comparar_periodo_comum(base), pasta, "comparacao_periodo_comum", True)
    salvar_tabela(da.media_mensal(base), pasta, "media_mensal", True)
    salvar_tabela(mv.tabela_ausentes(base), pasta, "ausentes")
    salvar_tabela(mv.padrao_card_pagamento(base), pasta, "card_nulo_x_pagamento", True)
    salvar_tabela(mv.cobertura_calendario(base), pasta, "cobertura_calendario", True)
    salvar_tabela(za.tabela_zeros(base), pasta, "dias_zerados")
    salvar_tabela(za.resumo_zeros(base), pasta, "resumo_zeros", True)
    salvar_tabela(oa.resumo_criterios(base, quebras), pasta, "outliers_criterios")
    atip = pd.concat([oa.investigar(base, s, a, quebras.get(s)) for s in ("Máquina 1", "Máquina 2", "Total") for a in ("demanda", "receita")])
    salvar_tabela(atip, pasta, "outliers_investigacao")
    salvar_tabela(oa.sensibilidade_modelagem(base), pasta, "outliers_sensibilidade", True)
    salvar_tabela(du.resumo_duplicados(base), pasta, "duplicados_resumo", True)
    salvar_tabela(du.duplicatas_exatas(base), pasta, "duplicados_exatos")
    salvar_tabela(du.pedidos_mesmo_segundo(base), pasta, "duplicados_mesmo_segundo")
    salvar_tabela(pd.DataFrame([du.recompras_rapidas(base)[1]]), pasta, "duplicados_recompras_m1")
    salvar_tabela(pa.concentracao(base), pasta, "produtos_concentracao", True)
    for s in ("Máquina 1", "Máquina 2", "Total"):
        salvar_tabela(pa.mix_produtos(base, s), pasta, f"produtos_mix_{CFG['series']['pastas'][s]}", True)
    salvar_tabela(pa.mudancas_preco(base), pasta, "mudancas_preco_m1")
    # visão de negócio: calendário das anomalias, concentração, insumos, ticket por público, padrão semanal
    cal_anom, cal_resumo = oa.calendario_anomalias(base, quebras)
    salvar_tabela(cal_anom, pasta, "outliers_calendario")
    salvar_tabela(cal_resumo, pasta, "outliers_calendario_resumo")
    salvar_tabela(pa.resumo_concentracao_negocio(base), pasta, "produtos_concentracao_negocio", True)
    for s in ("Máquina 1", "Máquina 2"):
        salvar_tabela(pa.curva_pareto(base, s), pasta, f"produtos_pareto_{CFG['series']['pastas'][s]}", True)
    salvar_tabela(pa.familias_insumo(base), pasta, "produtos_familias_insumo")
    salvar_tabela(pa.ticket_semanal(base), pasta, "ticket_semanal", True)
    quebra_2025 = quebras["Máquina 1"][-1]
    tickets, mix_m1, resumo_ticket = pa.ticket_por_publico(base, quebra_2025)
    salvar_tabela(tickets, pasta, "ticket_por_publico", True)
    salvar_tabela(mix_m1, pasta, "ticket_mix_m1_antes_depois", True)
    salvar_tabela(pd.DataFrame([{k: v for k, v in resumo_ticket.items() if not isinstance(v, dict)}]), pasta, "ticket_resumo")
    salvar_tabela(pd.Series(resumo_ticket["desconto da M2 nos mesmos produtos (%)"], name="M2 vs M1 (%)").rename_axis("produto").to_frame(),
                  pasta, "ticket_desconto_m2", True)
    perfil_sem, teste_sem = ts.perfil_semanal_antes_depois(base.series[("Máquina 1", "demanda")], quebras["Máquina 1"], quebra_2025)
    salvar_tabela(perfil_sem, pasta, "sazonalidade_semanal_m1_antes_depois", True)
    salvar_tabela(teste_sem, pasta, "sazonalidade_semanal_m1_teste")
    # estrutura temporal (notebook 05): só o treino da origem principal, com a ampliação do horário como regime
    from src.structural_breaks.structural_break_test import DetectorQuebras
    det = DetectorQuebras(base.series, base.transacoes)
    q05 = {s: [det.janela_ate(s, origem)] + det.quebras_ate(s, origem) for s in ("Máquina 1", "Total")}
    salvar_tabela(pd.DataFrame([ts.resumo_estrutura_temporal(base.series[(s, "demanda")].loc[:origem], q05[s], s) for s in q05]),
                  pasta, "tendencia_sazonalidade_resumo")
    salvar_tabela(ts.serie_degraus_semanal(base.series[("Máquina 1", "demanda")].loc[:origem], q05["Máquina 1"]), pasta,
                  "tendencia_degraus_m1_semanal", True)
    salvar_tabela(ts.teste_sazonalidade_semanal(base.series[("Máquina 1", "demanda")], q05["Máquina 1"])[0], pasta,
                  "sazonalidade_semanal_m1_regimes")
    # metodologia de teste.ipynb só com 2025 (série até o fim, como o teste.ipynb, e só o treino como sensibilidade)
    chaves = [(s, a) for s in ("Máquina 1", "Total", "Máquina 2") for a in ("demanda", "receita")]
    salvar_tabela(ts.fisher_recortes(base.series, chaves, fins=(None, origem)), pasta, "sazonalidade_fisher_2025")
    res_2025 = ts.metodologia_fisher(base.series[("Máquina 1", "demanda")].loc["2025-02-01":], "Máquina 1 — demanda desde 01/02/2025")
    salvar_tabela(ts.periodograma_etapa(res_2025), pasta, "periodograma_m1_2025")
    salvar_tabela(ts.perfil_semanal_fisher(res_2025).to_frame(), pasta, "perfil_fisher_m1_2025", True)
    for nome, fig in [("01_distribuicoes", da.grafico_distribuicoes(base)), ("02_evolucao", da.grafico_evolucao(base, rot["Total"])),
                      ("03_hora_dia_semana", da.grafico_hora_dia(base)), ("04_perfis", da.grafico_perfis(base)),
                      ("05_mensal", da.grafico_mensal(base)), ("06_ausentes", mv.grafico_ausentes(base)),
                      ("07_zeros", za.grafico_zeros(base)), ("08_outliers", oa.grafico_outliers(base, rot)),
                      ("09_mix_produtos", pa.grafico_mix(base)), ("10_evolucao_mix_precos", pa.grafico_evolucao_mix(base)),
                      ("11_ticket_medio", pa.grafico_ticket(base)), ("12_pareto_produtos", pa.grafico_pareto(base)),
                      ("13_perfil_semanal_m1", ts.grafico_perfil_antes_depois(perfil_sem, "Máquina 1")),
                      ("14_duplicados_recompras", du.grafico_recompras(base)),
                      ("15_fisher_m1_2025", ts.grafico_etapas_fisher(res_2025, cor_serie("Máquina 1")))]:
        salvar_figura(fig, pasta, nome)
    log.info("EDA: tabelas e gráficos em %s", pasta)


def etapa_quebras(base: BaseDados) -> None:
    from src.structural_breaks import cannibalization_analysis as cn, customer_analysis as ca, structural_break_test as sb

    pasta = pasta_relatorio("structural_breaks")
    dd = _datas(base)
    r = sb.bateria_testes(base.series, base.transacoes, dd["origem"])
    salvar_tabela(r["globais"], pasta, "testes_globais")
    salvar_tabela(r["segmentadas"], pasta, "regressao_segmentada")
    for s, (datas, tabela, _) in r["deteccao"].items():
        salvar_tabela(tabela, pasta, f"deteccao_bic_{CFG['series']['pastas'][s]}")
        salvar_tabela(r["locais"][s], pasta, f"confirmacao_local_{CFG['series']['pastas'][s]}")
    salvar_tabela(pd.Series(r["janela"]).to_frame("valor"), pasta, "quebra_janela_vendas", True)
    salvar_figura(sb.grafico_quebras(base.series, r, dd["origem"]), pasta, "01_quebras_nivel")
    janela, refinada = r["janela"]["data detectada (só treino)"], r["janela"]["data refinada (série inteira)"]
    salvar_figura(sb.grafico_janela(base.transacoes, janela, refinada), pasta, "02_janela_vendas")

    # 6.2 Novos clientes (quebra final)
    quebra_final = r["detector"].quebra_novos_clientes_ate("Máquina 1", dd["origem"])
    sem = ca.evolucao_semanal(base.transacoes)
    dec = ca.decompor_crescimento(base.transacoes, quebra_final, base.fim)
    afluxo = ca.teste_afluxo_novos(sem, quebra_final)
    coortes = ca.frequencia_e_retencao(base.transacoes, quebra_final, base.fim)
    freq = ca.comparar_frequencia(base.transacoes, quebra_final, base.fim)
    salvar_tabela(sem, pasta, "clientes_semanal", True)
    salvar_tabela(dec, pasta, "clientes_decomposicao_crescimento", True)
    salvar_tabela(coortes, pasta, "clientes_coortes")
    salvar_tabela(pd.DataFrame([afluxo, freq]).T, pasta, "clientes_testes", True)
    salvar_figura(ca.grafico_clientes(sem, dec, quebra_final), pasta, "03_novos_clientes")

    # 6.3 Canibalização
    horario = cn.analisar_mudanca_horario(base.transacoes, refinada)
    cartoes = cn.deslocamento_cartoes(base.transacoes, refinada)
    m2 = cn.analisar_entrada_m2(base.transacoes, base.series, base.inicio["Máquina 2"], quebra_final, base.fim)
    sint = cn.sintese_evidencias(horario, cartoes, m2)
    salvar_tabela(horario, pasta, "canibalizacao_horario_its", True)
    salvar_tabela(pd.Series(cartoes).to_frame("valor"), pasta, "canibalizacao_horario_cartoes", True)
    salvar_tabela(m2["composicao"], pasta, "canibalizacao_m2_faixas", True)
    salvar_tabela(pd.Series({**m2["nivel"], **m2["correlacao"], **m2["fim_de_semana"], "χ² composição p": m2["qui_quadrado_p"]}).to_frame("valor"),
                  pasta, "canibalizacao_m2_testes", True)
    salvar_tabela(sint, pasta, "canibalizacao_sintese")
    salvar_figura(cn.grafico_mudanca_horario(base.transacoes, refinada), pasta, "04_canibalizacao_horario")
    salvar_figura(cn.grafico_entrada_m2(m2, base.series, base.inicio["Máquina 2"], quebra_final), pasta, "05_canibalizacao_entrada_m2")

    datas_dummies = pd.DataFrame([
        {"dummy": "dummy_horario_reduzido", "definição": "1 antes da extensão da janela de vendas",
         "data (só treino)": janela, "data (série inteira)": refinada,
         "evidência": f"Binseg+BIC na % fora de 10–19h ({r['janela']['% fora 10–19h antes']:.0f}% → {r['janela']['% fora 10–19h depois']:.0f}%); "
                      f"sup-Wald p < 0,001; sup-Wald global da demanda da M1 estima a maior quebra em "
                      f"{r['globais'].iloc[0]['detalhe'].split('data estimada ')[1].split(';')[0]}"},
        {"dummy": "dummy_novos_clientes", "definição": "1 a partir do aumento de patamar de 2025",
         "data (só treino)": quebra_final, "data (série inteira)": r["detector"].quebra_novos_clientes_ate("Máquina 1", base.fim),
         "evidência": f"Binseg+BIC (M1 {quebra_final:%d/%m/%Y}; Total {r['detector'].quebra_novos_clientes_ate('Total', dd['origem']):%d/%m/%Y}); "
                      f"{dec.attrs['cartoes_novos']} de {dec.attrs['cartoes_depois']} cartões ativos depois são novos"},
    ])
    salvar_tabela(datas_dummies, pasta, "datas_dummies")
    salvar_artefato("quebras", {"testes": {k: v for k, v in r.items() if k != "detector"}, "clientes": {"semanal": sem, "decomposicao": dec,
                    "afluxo": afluxo, "coortes": coortes, "frequencia": freq},
                    "canibalizacao": {"horario": horario, "cartoes": cartoes, "m2": m2, "sintese": sint},
                    "datas_dummies": datas_dummies})
    log.info("Quebras: janela %s (refinada %s), quebra final %s", f"{janela:%d/%m/%Y}", f"{refinada:%d/%m/%Y}", f"{quebra_final:%d/%m/%Y}")


def etapa_features(base: BaseDados) -> None:
    import matplotlib.pyplot as plt

    from src.features import feature_pipeline as fp
    from src.structural_breaks.structural_break_test import DetectorQuebras
    from src.visualization.plots import CORES

    pasta = pasta_relatorio("feature_analysis")
    dd = _datas(base)
    det = DetectorQuebras(base.series, base.transacoes)
    linhas, contrib, imps, cors = [], [], [], {}
    for s in CFG["series"]["nomes"]:
        c = fp.ConstrutorAtributos(base, det, s, dd["origem"])
        y = base.series[(s, "demanda")]
        X = c(y.loc[:dd["origem"]])
        linhas.append({"série": s, "atributos": X.shape[1], "linhas de treino": X.shape[0],
                       "teste: futuro substituído por ruído não altera atributos": fp.teste_vazamento_futuro(c, y),
                       "teste: recursão com cauda = série inteira": fp.teste_consistencia_recursao(c, y.loc[:dd["origem"]])})
        cors[s] = fp.correlacao_com_alvo(X, y.loc[:dd["origem"]])
        if s != "Máquina 2":
            contrib.append(fp.contribuicao_dummies(base, det, s, dd["origem"]).assign(série=s))
            for a in CFG["series"]["alvos"]:
                imps.append(fp.importancia_permutacao(base, det, s, a, dd["origem"] - pd.Timedelta(days=42), dd["origem"]).assign(série=s, alvo=a))
        if s == "Máquina 1":
            salvar_tabela(X.tail(20), pasta, "exemplo_matriz_atributos_m1", True)
    vaz = pd.DataFrame(linhas)
    contrib, imps = pd.concat(contrib), pd.concat(imps)
    salvar_tabela(vaz, pasta, "testes_vazamento")
    salvar_tabela(contrib, pasta, "contribuicao_dummies")
    salvar_tabela(imps, pasta, "importancia_permutacao")
    salvar_tabela(pd.DataFrame(cors), pasta, "correlacao_spearman_demanda", True)
    if not vaz.iloc[:, 3:].all().all():
        raise AssertionError("Teste de vazamento falhou: ver reports/feature_analysis/testes_vazamento.csv")

    fig, eixos = plt.subplots(1, 2, figsize=(16, 4.8))
    for ax, s in zip(eixos, ("Máquina 1", "Total")):
        t = imps[(imps["série"] == s) & (imps["alvo"] == "demanda")].set_index("grupo").sort_values("aumento relativo %")
        ax.barh(t.index, t["aumento relativo %"],
                color=[cor_serie(s) if v > 0 else CORES["contexto"] for v in t["aumento relativo %"]], height=0.6)
        ax.axvline(0, color=CORES["eixo"], lw=1)
        ax.set_xlabel("aumento do MAE ao permutar o grupo (%)")
        ax.set_title(f"{s} — demanda: importância por permutação (fora da amostra)")
    fig.tight_layout()
    salvar_figura(fig, pasta, "01_importancia_permutacao")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for s, desloc in (("Máquina 1", -0.2), ("Total", 0.2)):
        t = contrib[contrib["série"] == s]
        ax.bar(np.arange(len(t)) + desloc, -t["ΔAIC vs só dia da semana"], width=0.38, color=cor_serie(s), label=s)
    ax.set_xticks(np.arange(len(t)))
    ax.set_xticklabels(t["modelo"], rotation=12, fontsize=8.5)
    ax.set_ylabel("redução do AIC vs só dia da semana")
    ax.set_title("Contribuição das dummies de quebra (NB2 com dia da semana, treino)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    salvar_figura(fig, pasta, "02_contribuicao_dummies")
    fig, ax = plt.subplots(figsize=(10, 6))
    top = cors["Máquina 1"].head(18)[::-1]
    ax.barh(top.index, top.values, color=[cor_serie("Máquina 1") if v > 0 else CORES["Máquina 2"] for v in top.values], height=0.65)
    ax.axvline(0, color=CORES["eixo"], lw=1)
    ax.set_xlabel("ρ de Spearman com a demanda diária (treino)")
    ax.set_title("Máquina 1: atributos mais associados à demanda")
    fig.tight_layout()
    salvar_figura(fig, pasta, "03_correlacao_atributos_m1")
    salvar_artefato("features", {"vazamento": vaz, "contribuicao": contrib, "importancia": imps, "correlacao": cors})


def etapa_modelagem(base: BaseDados) -> list:
    from src.validation.backtesting import executar_origem

    dd = _datas(base)
    t0 = time.time()
    resultados, nao = executar_origem(base, dd["origem"], SERIES, cenario="principal")
    salvar_artefato("resultados_principal", {"resultados": resultados, "nao_aplicaveis": nao})
    log.info("Modelagem na origem principal: %d resultados em %.0f s", len(resultados), time.time() - t0)
    return resultados


def etapa_backtest(base: BaseDados) -> None:
    from src.validation.backtesting import backtest, tabela_resultados

    dd = _datas(base)
    series_bt = [(s, a) for s in CFG["divisao"]["series_backtest"] for a in CFG["series"]["alvos"]]
    resultados, nao = backtest(base, dd["origens_backtest"], series_bt)
    tab = tabela_resultados(resultados, base)
    salvar_artefato("backtest", {"tabela": tab, "nao_aplicaveis": nao,
                                 "previsoes": {(r.modelo, r.serie, r.alvo, r.origem): r.previsao for r in resultados}})


def etapa_avaliacao(base: BaseDados) -> dict:
    from src.validation import model_comparison as mc
    from src.validation.backtesting import tabela_resultados

    pasta = pasta_relatorio("model_comparison")
    princ = carregar_artefato("resultados_principal")["resultados"]
    tab_teste = tabela_resultados(princ, base)
    try:
        bt = carregar_artefato("backtest")["tabela"]
        principal_bt = tab_teste[tab_teste["série"].isin(CFG["divisao"]["series_backtest"])]
        bt_completo = pd.concat([bt, principal_bt], ignore_index=True)
        resumo_bt = mc.resumo_backtest(bt_completo)
    except FileNotFoundError:
        log.warning("Backtest indisponível (modo rápido): a seleção usa só teste e validação")
        bt_completo, resumo_bt = None, None
    dm = mc.tabela_dm(princ, base)
    dm_m4 = mc.tabela_dm(princ, base, "Média 4 semanas")
    bu = mc.comparar_bottom_up(princ, base)
    tvt = mc.tabela_treino_validacao_teste(princ, base, resumo_bt)
    sel = mc.selecao_final(tab_teste, resumo_bt)
    escolhidos = sel[sel["escolhido"]]
    residuos = pd.DataFrame([mc.diagnostico_residuos(r, base) for r in princ])
    salvar_tabela(tab_teste.drop(columns=["cenário"]), pasta, "metricas_teste")
    salvar_tabela(dm, pasta, "diebold_mariano_vs_naive")
    salvar_tabela(dm_m4, pasta, "diebold_mariano_vs_media4")
    salvar_tabela(bu, pasta, "total_direto_vs_soma_maquinas")
    salvar_tabela(tvt, pasta, "treino_validacao_teste")
    salvar_tabela(sel, pasta, "selecao_final")
    salvar_tabela(residuos, pasta, "diagnostico_residuos")
    salvar_figura(mc.grafico_heatmap_teste(tab_teste), pasta, "01_teste_wape_vies")
    salvar_figura(mc.grafico_bottom_up(bu), pasta, "04_total_direto_vs_soma")
    salvar_figura(mc.grafico_treino_validacao_teste(tvt), pasta, "05_treino_validacao_teste")
    idx = {(r.modelo, r.serie, r.alvo): r for r in princ}
    for _, e in escolhidos.iterrows():
        r = idx[(e["modelo"], e["série"], e["alvo"])]
        salvar_figura(mc.grafico_residuos(r, base), pasta / "residuos",
                      f"residuos_{CFG['series']['pastas'][r.serie]}_{r.alvo}")
    if resumo_bt is not None:
        salvar_tabela(bt_completo, pasta, "backtest_metricas_por_origem")
        salvar_tabela(resumo_bt, pasta, "backtest_resumo")
        salvar_tabela(mc.erros_por_periodo(bt_completo), pasta, "backtest_erro_por_periodo", True)
        salvar_figura(mc.grafico_backtest(resumo_bt), pasta, "02_backtest_mase")
        for s, a in [(s, a) for s in CFG["divisao"]["series_backtest"] for a in CFG["series"]["alvos"]]:
            salvar_figura(mc.grafico_backtest_origens(bt_completo, s, a), pasta, f"03_backtest_origens_{CFG['series']['pastas'][s]}_{a}")
    saida = {"tab_teste": tab_teste, "resumo_bt": resumo_bt, "bt": bt_completo, "dm": dm, "dm_m4": dm_m4, "bottom_up": bu,
             "tvt": tvt, "selecao": sel, "residuos": residuos}
    salvar_artefato("avaliacao", saida)
    log.info("Modelos escolhidos: %s", {f"{r['série']}/{r['alvo']}": r["modelo"] for _, r in escolhidos.iterrows()})
    return saida


def grafico_previsoes_maquinas(base: BaseDados, dias_historico: int = 35) -> None:
    """Previsão final de 14 dias por máquina: Máquina 1 em cima, Máquina 2 embaixo (demanda e receita).

    Lê ``previsao_futura_14_dias.csv`` de cada máquina, então pode ser refeita sem rodar os modelos.
    """
    import matplotlib.pyplot as plt

    fig, eixos = plt.subplots(2, 2, figsize=(17, 9))
    for i, serie in enumerate(("Máquina 1", "Máquina 2")):
        fut = pd.read_csv(pasta_relatorio("forecasts", CFG["series"]["pastas"][serie]) / "previsao_futura_14_dias.csv", parse_dates=["data"])
        for j, alvo in enumerate(CFG["series"]["alvos"]):
            f = fut[fut["alvo"] == alvo].set_index("data")
            y = base.series[(serie, alvo)].dropna()
            plot_previsao(eixos[i, j], y.loc[base.fim - pd.Timedelta(days=dias_historico - 1):], None, f["previsto"],
                          f[["li_80", "ls_80", "li_95", "ls_95"]], cor_serie(serie),
                          f"{serie} — {alvo}: {f['modelo'].iloc[0]}", CFG["series"]["unidades"][alvo], "Previsão")
    fig.suptitle(f"Previsão dos próximos 14 dias ({fut['data'].min():%d/%m}–{fut['data'].max():%d/%m/%Y}): Máquina 1 (acima) e Máquina 2 (abaixo)")
    fig.tight_layout()
    salvar_figura(fig, pasta_relatorio("forecasts"), "previsoes_14_dias_m1_m2")


def tabela_qualidade_escolhidos() -> pd.DataFrame:
    """Métricas de qualidade no teste dos modelos escolhidos e da soma M1 + M2 (recomendação para o Total).

    Lê as previsões de teste já gravadas, então pode ser refeita sem rodar os modelos. A soma não tem
    intervalo próprio (somar limites não dá a cobertura certa), então fica sem cobertura.
    """
    from src.validation.metrics import calcular_metricas

    bt = pd.read_csv(pasta_relatorio("model_comparison") / "backtest_resumo.csv")
    diag = pd.read_csv(pasta_relatorio("model_comparison") / "diagnostico_residuos.csv")
    linhas, prev = [], {}
    for serie in CFG["series"]["nomes"]:
        t = pd.read_csv(pasta_relatorio("forecasts", CFG["series"]["pastas"][serie]) / "previsoes_teste_todos_modelos.csv", parse_dates=["data"])
        for alvo, g in t[t["escolhido"]].groupby("alvo"):
            g = g.set_index("data")
            prev[(serie, alvo)] = g
            m = calcular_metricas(g["observado"], g["previsto"], intervalos=g[["li_80", "ls_80", "li_95", "ls_95"]])
            b = bt[(bt["série"] == serie) & (bt["alvo"] == alvo) & (bt["modelo"] == g["modelo"].iloc[0])]
            d = diag[(diag["série"] == serie) & (diag["alvo"] == alvo) & (diag["modelo"] == g["modelo"].iloc[0])]
            linhas.append({"série": serie, "alvo": alvo, "modelo": g["modelo"].iloc[0], **m,
                           "MASE backtest": b["MASE médio"].iloc[0] if len(b) else np.nan,
                           "Ljung-Box p (erros de teste)": d["Ljung-Box p (erros de teste)"].iloc[0] if len(d) else np.nan})
    for alvo in CFG["series"]["alvos"]:
        m1, m2 = prev[("Máquina 1", alvo)], prev[("Máquina 2", alvo)]
        soma = m1["previsto"] + m2["previsto"].reindex(m1.index).fillna(0)
        obs = m1["observado"] + m2["observado"].reindex(m1.index).fillna(0)
        linhas.append({"série": "Total", "alvo": alvo, "modelo": "Soma M1 + M2", **calcular_metricas(obs, soma)})
    return pd.DataFrame(linhas)


def etapa_previsoes(base: BaseDados) -> None:
    import matplotlib.pyplot as plt

    from src.models.model_selection import Contexto
    from src.structural_breaks.structural_break_test import DetectorQuebras
    from src.validation.backtesting import executar_modelo
    from src.validation.metrics import calcular_metricas

    princ = carregar_artefato("resultados_principal")["resultados"]
    sel = carregar_artefato("avaliacao")["selecao"]
    escolhidos = {(r["série"], r["alvo"]): r["modelo"] for _, r in sel[sel["escolhido"]].iterrows()}
    det = DetectorQuebras(base.series, base.transacoes)
    pasta_modelos = caminho("modelos")
    finais, registro = {}, []
    for (serie, alvo), modelo in escolhidos.items():
        ctx = Contexto(base, det, serie, alvo, base.fim, "final", CFG["divisao"]["horizonte_futuro"])
        r = executar_modelo(modelo, ctx)
        finais[(serie, alvo)] = r
        nome = f"{CFG['series']['pastas'][serie]}_{alvo}"
        objeto = r.extras.get("objeto_modelo")
        if objeto is not None:
            joblib.dump(objeto, pasta_modelos / f"{nome}.joblib", compress=3)
        meta = {"série": serie, "alvo": alvo, "modelo": modelo, "especificação": r.especificacao,
                "treino": [f"{r.inicio_treino:%Y-%m-%d}", f"{base.fim:%Y-%m-%d}"], "horizonte": ctx.horizonte,
                "previsão": [f"{ctx.datas_previsao[0]:%Y-%m-%d}", f"{ctx.datas_previsao[-1]:%Y-%m-%d}"],
                "hash_config": hash_config(), "hash_dados": json.loads((caminho("dados_brutos") / "MANIFEST.json").read_text())["arquivos"],
                "arquivo_modelo": f"{nome}.joblib" if objeto is not None else None,
                "gerado_em": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
        (pasta_modelos / f"{nome}_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        registro.append(meta)
    pd.DataFrame(registro).to_csv(pasta_modelos / "registro_modelos.csv", index=False)

    idx = {(r.modelo, r.serie, r.alvo): r for r in princ}
    for serie in CFG["series"]["nomes"]:
        pasta = pasta_relatorio("forecasts", CFG["series"]["pastas"][serie])
        linhas_teste, linhas_futuro, metricas = [], [], []
        fig, eixos = plt.subplots(2, 2, figsize=(17, 9))
        for j, alvo in enumerate(CFG["series"]["alvos"]):
            y = base.series[(serie, alvo)]
            modelo = escolhidos[(serie, alvo)]
            for (m, s, a), r in idx.items():
                if s == serie and a == alvo:
                    obs = y.loc[r.previsao.index]
                    df = pd.DataFrame({"data": r.previsao.index, "modelo": m, "alvo": alvo, "observado": obs.values,
                                       "previsto": r.previsao.values, "escolhido": m == modelo})
                    if r.intervalos is not None:
                        df = df.join(r.intervalos.reset_index(drop=True))
                    linhas_teste.append(df)
                    metricas.append({"modelo": m, "alvo": alvo, "escolhido": m == modelo,
                                     **calcular_metricas(obs, r.previsao, y.loc[:r.origem], r.intervalos)})
            rt, rf = idx[(modelo, serie, alvo)], finais[(serie, alvo)]
            unidade = CFG["series"]["unidades"][alvo]
            plot_previsao(eixos[0, j], y.loc[rt.origem - pd.Timedelta(days=27):rt.origem], y.loc[rt.previsao.index], rt.previsao,
                          rt.intervalos, cor_serie(serie), f"{serie} — {alvo}: teste ({modelo})", unidade)
            plot_previsao(eixos[1, j], y.loc[base.fim - pd.Timedelta(days=27):], None, rf.previsao, rf.intervalos, cor_serie(serie),
                          f"{serie} — {alvo}: previsão {rf.previsao.index[0]:%d/%m}–{rf.previsao.index[-1]:%d/%m/%Y}", unidade,
                          "Previsão final")
            fut = pd.DataFrame({"data": rf.previsao.index, "alvo": alvo, "modelo": modelo, "previsto": rf.previsao.values})
            if rf.intervalos is not None:
                fut = fut.join(rf.intervalos.reset_index(drop=True))
            linhas_futuro.append(fut)
        fig.suptitle(f"{serie}: previsões do modelo escolhido — teste (acima) e próximos 14 dias (abaixo)")
        fig.tight_layout()
        salvar_figura(fig, pasta, "previsoes_modelo_escolhido")
        salvar_tabela(pd.concat(linhas_teste), pasta, "previsoes_teste_todos_modelos")
        salvar_tabela(pd.concat(linhas_futuro), pasta, "previsao_futura_14_dias")
        salvar_tabela(pd.DataFrame(metricas).sort_values(["alvo", "MAE"]), pasta, "metricas_teste")

    grafico_previsoes_maquinas(base)
    salvar_tabela(tabela_qualidade_escolhidos(), pasta_relatorio("model_comparison"), "qualidade_modelos_escolhidos")

    # Total: modelo independente × soma das previsões individuais (modelos escolhidos de cada máquina)
    pasta = pasta_relatorio("forecasts", "total")
    comp, fig = [], plt.figure(figsize=(16, 4.8))
    eixos = fig.subplots(1, 2)
    for ax, alvo in zip(eixos, CFG["series"]["alvos"]):
        y = base.series[("Total", alvo)]
        direto = idx[(escolhidos[("Total", alvo)], "Total", alvo)].previsao
        soma = idx[(escolhidos[("Máquina 1", alvo)], "Máquina 1", alvo)].previsao + idx[(escolhidos[("Máquina 2", alvo)], "Máquina 2", alvo)].previsao
        obs = y.loc[direto.index]
        fut_d = finais[("Total", alvo)].previsao
        fut_s = finais[("Máquina 1", alvo)].previsao + finais[("Máquina 2", alvo)].previsao
        for rot, p in (("Total direto", direto), ("Soma M1 + M2", soma)):
            comp.append({"alvo": alvo, "abordagem": rot, **calcular_metricas(obs, p, y.loc[:direto.index[0] - pd.Timedelta(days=1)])})
        comp.append({"alvo": alvo, "abordagem": "diferença média futura (direto − soma)", "ME": float((fut_d - fut_s).mean())})
        ax.plot(obs.index, obs.values, color="#2b2b2b", marker="o", ms=4, label="Observado (teste)")
        ax.plot(direto.index, direto.values, color=cor_serie("Total"), lw=2, ls="--", marker="o", ms=4, label="Total direto")
        ax.plot(soma.index, soma.values, color=cor_serie("Máquina 1"), lw=2, ls=":", marker="s", ms=4, label="Soma M1 + M2")
        ax.plot(fut_d.index, fut_d.values, color=cor_serie("Total"), lw=2, ls="--")
        ax.plot(fut_s.index, fut_s.values, color=cor_serie("Máquina 1"), lw=2, ls=":")
        ax.axvline(base.fim + pd.Timedelta(hours=12), color="#52514e", ls="--", lw=1)
        ax.annotate("futuro", xy=(base.fim + pd.Timedelta(days=1), 0.97), xycoords=("data", "axes fraction"), fontsize=8.5, va="top")
        ax.set_title(f"Total — {alvo}: modelo independente × soma das máquinas")
        ax.set_ylabel(CFG["series"]["unidades"][alvo])
        from src.visualization.plots import formatar_eixo_datas, reservar_topo
        reservar_topo(ax, 0.3)                       # legenda e rótulo "futuro" acima das linhas
        ax.legend(loc="upper left", fontsize=8)     # 3 linhas à esquerda; "futuro" fica à direita da linha vertical
        formatar_eixo_datas(ax, curto=True)
    fig.tight_layout()
    salvar_figura(fig, pasta, "total_direto_vs_soma_maquinas")
    salvar_tabela(pd.DataFrame(comp), pasta, "total_direto_vs_soma_maquinas")
    salvar_artefato("previsoes_finais", {"finais": finais, "escolhidos": escolhidos, "comparacao_total": pd.DataFrame(comp)})
    log.info("Previsões finais gravadas em reports/forecasts e modelos em models/")


# ============================================================================ orquestração
def executar(etapas: list[str] | None = None, rapido: bool = False) -> None:
    """Executa as etapas na ordem; cada etapa registra o tempo e erros no log."""
    etapas = etapas or ETAPAS
    if rapido and "backtest" in etapas:
        etapas = [e for e in etapas if e != "backtest"]
        arquivo = pasta_artefatos() / "backtest.pkl"
        if arquivo.exists():
            arquivo.unlink()
    log.info("Pipeline iniciado: etapas = %s | config %s", etapas, hash_config())
    t_total = time.time()
    base = etapa_dados() if "dados" in etapas else carregar_artefato("base")
    funcoes = {"identificacao": etapa_identificacao, "eda": etapa_eda, "quebras": etapa_quebras, "features": etapa_features,
               "modelagem": etapa_modelagem, "backtest": etapa_backtest, "avaliacao": etapa_avaliacao, "previsoes": etapa_previsoes}
    for etapa in [e for e in ETAPAS if e in etapas and e != "dados"]:
        t0 = time.time()
        try:
            funcoes[etapa](base)
        except Exception:
            log.exception("Falha na etapa '%s'", etapa)
            raise
        log.info("Etapa '%s' concluída em %.0f s", etapa, time.time() - t0)
    log.info("Pipeline concluído em %.1f min", (time.time() - t_total) / 60)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")      # execução sem tela; nos notebooks o backend inline é preservado
    parser = argparse.ArgumentParser(description="Pipeline Coffee Sales")
    parser.add_argument("--etapas", nargs="+", choices=ETAPAS, help="etapas a executar (padrão: todas)")
    parser.add_argument("--rapido", action="store_true", help="pula o backtest (seleção sem as 11 origens)")
    args = parser.parse_args()
    executar(args.etapas, args.rapido)


if __name__ == "__main__":
    main()
