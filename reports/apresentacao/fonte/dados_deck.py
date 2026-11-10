"""Extrai dos relatórios do pipeline os números e as imagens usados na apresentação (dados_deck.json).

Uso: python dados_deck.py  (depois de ``python -m src.pipeline``). Nenhum número do deck é digitado à mão
quando existe uma tabela em reports/ que o contenha.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd
from PIL import Image

R = Path(__file__).resolve().parents[2]          # .../case_coffee_sales/reports
SAIDA = Path(__file__).with_name("dados_deck.json")
E = R / "eda"


def registros(df: pd.DataFrame, casas: int = 2) -> list[dict]:
    return df.round(casas).astype(object).where(df.notna(), None).to_dict(orient="records")


imagens = {
    "perfis": "structural_breaks/identificacao_maquinas/perfis_periodo_comum.png",
    "evolucao": "eda/02_evolucao.png",
    "quebras": "structural_breaks/01_quebras_nivel.png",
    "clientes": "structural_breaks/03_novos_clientes.png",
    "canib_horario": "structural_breaks/04_canibalizacao_horario.png",
    "canib_m2": "structural_breaks/05_canibalizacao_entrada_m2.png",
    "backtest": "model_comparison/02_backtest_mase.png",
    "prev_m1_m2": "forecasts/previsoes_14_dias_m1_m2.png",
    "janela": "structural_breaks/02_janela_vendas.png",
    "total_soma": "forecasts/total/total_direto_vs_soma_maquinas.png",
}
saida = {"imagens": {}}
for k, rel in imagens.items():
    p = R / rel
    w, h = Image.open(p).size
    # caminho relativo a esta pasta: o deck é gerado em qualquer máquina (o gerador resolve a partir de __dirname)
    saida["imagens"][k] = {"path": os.path.relpath(p, SAIDA.parent).replace(os.sep, "/"), "ratio": w / h}

# ------------------------------------------------------------------ modelos
sel = pd.read_csv(R / "model_comparison/selecao_final.csv")
bt = pd.read_csv(R / "model_comparison/backtest_resumo.csv")
linhas = []
for _, r in sel[sel["escolhido"]].iterrows():
    b = bt[(bt["série"] == r["série"]) & (bt["alvo"] == r["alvo"]) & (bt["modelo"] == r["modelo"])]
    ref = sel[(sel["série"] == r["série"]) & (sel["alvo"] == r["alvo"])].set_index("modelo")
    linhas.append({"serie": r["série"], "alvo": r["alvo"], "modelo": r["modelo"], "wape": round(r["WAPE_%"], 1), "mae": round(r["MAE"], 2),
                   "vies": round(r["viés relativo %"], 1), "mae_vs_naive": round(100 * (r["MAE"] / ref.loc["Naive sazonal", "MAE"] - 1), 0),
                   "wape_media4": round(ref.loc["Média 4 semanas", "WAPE_%"], 1),
                   "mase_bt": None if b.empty else round(b["MASE médio"].iloc[0], 3),
                   "p_holm_naive": None if b.empty or pd.isna(b["p Holm vs naive"].iloc[0]) else round(b["p Holm vs naive"].iloc[0], 3)})
saida["escolhidos"] = linhas
dm = pd.read_csv(R / "model_comparison/diebold_mariano_vs_naive.csv")
saida["dm_m2_p_min"] = round(dm[dm["série"] == "Máquina 2"]["p"].min(), 2)
tvt = pd.read_csv(R / "model_comparison/treino_validacao_teste.csv")
saida["piso_ruido"] = {s: round(g["piso de ruído %"].iloc[0], 0) for s, g in tvt[tvt["alvo"] == "demanda"].groupby("série")}
bu = pd.read_csv(R / "forecasts/total/total_direto_vs_soma_maquinas.csv")
saida["total_soma"] = registros(bu)
fut = {}
for pasta in ("machine_1", "machine_2", "total"):
    f = pd.read_csv(R / f"forecasts/{pasta}/previsao_futura_14_dias.csv")
    fut[pasta] = {a: {"media": round(g["previsto"].mean(), 1), "soma": round(g["previsto"].sum(), 0), "modelo": g["modelo"].iloc[0],
                      "inicio": g["data"].min(), "fim": g["data"].max()} for a, g in f.groupby("alvo")}
fut["soma"] = {a: {"media": round(fut["machine_1"][a]["media"] + fut["machine_2"][a]["media"], 1),
                   "soma": fut["machine_1"][a]["soma"] + fut["machine_2"][a]["soma"]} for a in ("demanda", "receita")}
saida["futuro"] = fut

# ------------------------------------------------------------------ negócio: padrão semanal, ticket, produtos, anomalias
perfil = pd.read_csv(E / "sazonalidade_semanal_m1_antes_depois.csv", index_col=0)
teste = pd.read_csv(E / "sazonalidade_semanal_m1_teste.csv")
saida["semanal_m1"] = {"dias": list(perfil.columns), "periodos": list(perfil.index),
                       "valores": [[round(v, 1) for v in linha] for linha in perfil.to_numpy()], "testes": registros(teste, 4)}
comp = pd.read_csv(E / "comparacao_periodo_comum.csv", index_col=0)
saida["fds_m2_pct"] = round(comp.loc["Máquina 2", "% demanda no fim de semana"], 0)

ts = pd.read_csv(E / "ticket_semanal.csv", parse_dates=["data"])
saida["ticket_semanal"] = {"rotulos": [f"{d:%d/%m/%y}" for d in ts["data"]],
                           "m1": [None if pd.isna(v) else round(v, 2) for v in ts["Máquina 1"]],
                           "m2": [None if pd.isna(v) else round(v, 2) for v in ts["Máquina 2"]]}
tk = pd.read_csv(E / "ticket_por_publico.csv", index_col=0)
saida["ticket_publico"] = {g: {"ticket": round(r["ticket médio"], 1), "noite": round(r.iloc[-1], 0)} for g, r in tk.iterrows()}
rt = pd.read_csv(E / "ticket_resumo.csv").iloc[0]
saida["ticket_resumo"] = {"var_m1": round(rt["variação do ticket da M1 (%)"], 1), "m2_vs_m1": round(rt["M2 vs M1 no período comum (%)"], 0),
                          "ultima_tabela": str(rt["última mudança de tabela da M1"])[:10]}
mix = pd.read_csv(E / "ticket_mix_m1_antes_depois.csv", index_col=0)
saida["mix_m1"] = {p: {"antes": round(r["% antes"], 0), "depois": round(r["% depois"], 0), "preco": round(r["preço (u.m.)"], 1)} for p, r in mix.iterrows()}
desc = pd.read_csv(E / "ticket_desconto_m2.csv", index_col=0).iloc[:, 0]
saida["desconto_m2"] = {p: round(v, 0) for p, v in desc.items()}

conc = pd.read_csv(E / "produtos_concentracao_negocio.csv", index_col=0)
saida["concentracao"] = {s: {k: (int(v) if float(v).is_integer() else round(v, 2)) for k, v in r.items()} for s, r in conc.iterrows()}
mt = pd.read_csv(E / "produtos_mix_total.csv")
top = mt.head(8)
saida["receita_produtos_total"] = {"produtos": list(top["produto"]) + [f"Outros {len(mt) - 8} produtos (só M2)"],
                                   "pct": [round(v, 1) for v in top["% receita"]] + [round(mt["% receita"].iloc[8:].sum(), 1)]}
fam = pd.read_csv(E / "produtos_familias_insumo.csv")
saida["familias"] = {f"{r['série']}|{r['família']}": {"produtos": int(r["produtos"]), "vendas": int(r["vendas"]),
                                                      "pct_vendas": round(r["% vendas"], 2), "pct_receita": round(r["% receita"], 2)} for _, r in fam.iterrows()}
cal = pd.read_csv(E / "outliers_calendario_resumo.csv").set_index("sentido")
saida["anomalias"] = {s: {"dias": int(r["dias"]), "k": int(r["em feriado ou fim de semana"]), "pct": round(r["%"], 0),
                          "esperado": round(r["% esperado pelo calendário"], 0), "p": round(r["p (binomial, unilateral)"], 3)} for s, r in cal.iterrows()}
dup = pd.read_csv(E / "duplicados_resumo.csv", index_col=0).loc["Máquina 2"]
rr = pd.read_csv(E / "duplicados_recompras_m1.csv").iloc[0]
saida["duplicados"] = {"exatas_m2": int(dup["duplicatas exatas (registros a mais)"]), "grupos_m2": int(dup["grupos no mesmo segundo"]),
                       "grupos_dif_m2": int(dup["grupos com produtos diferentes"]), "pct_m2": round(dup["% das vendas"], 2),
                       "recompras_m1": int(rr["recompras do mesmo produto"]), "pct_m1": round(rr["% das vendas da M1"], 2),
                       "min_mesmo": round(rr["intervalo mínimo, mesmo produto (s)"], 1), "med_mesmo": round(rr["intervalo mediano, mesmo produto (s)"], 1),
                       "med_dif": round(rr["intervalo mediano, produto diferente (s)"], 1), "abaixo30": int(rr["recompras com intervalo < 30 s"])}
tr = pd.read_csv(E / "tendencia_sazonalidade_resumo.csv").set_index("série")
saida["tendencia"] = {s: {"dbic": round(r["ΔBIC reta − degraus"], 1), "regimes_incl": int(r["regimes com inclinação significativa"]),
                          "regimes": int(r["regimes"]), "picos": r["Fisher no ano inteiro: períodos dos picos (dias)"],
                          "p_degraus": round(r["Fisher com degraus: p"], 3), "p_banda": round(r["Fisher 2–31 dias após degraus: p"], 3),
                          "p_ciclo": round(r["ciclo: Fisher no desvio semanal (p)"], 3), "p_lb": round(r["ciclo: Ljung-Box p mínimo"], 3),
                          "p_dia_mes": round(r["dia do mês: p Kruskal-Wallis"], 3), "fim": str(r["fim"])[:10],
                          "incl_ano": round(r["inclinação no ano inteiro (% por semana)"], 2)} for s, r in tr.iterrows()}
dg = pd.read_csv(E / "tendencia_degraus_m1_semanal.csv", parse_dates=["data"])
saida["degraus_m1"] = {"rotulos": [f"{d:%d/%m/%y}" for d in dg["data"]], "demanda": [round(v, 2) for v in dg["demanda"]],
                       "nivel": [round(v, 2) for v in dg["nível do regime"]], "reta": [round(v, 2) for v in dg["reta"]]}
sr = pd.read_csv(E / "sazonalidade_semanal_m1_regimes.csv")
saida["semanal_regimes"] = {"r2_ultimo": round(sr["R² do dia da semana"].iloc[-1], 3), "p_ultimo": sr["p"].iloc[-1],
                            "r2_antes_min": round(sr["R² do dia da semana"].iloc[:-1].min(), 3), "r2_antes_max": round(sr["R² do dia da semana"].iloc[:-1].max(), 3),
                            "p_antes_min": round(sr["p"].iloc[:-1].min(), 3)}
pg = pd.read_csv(E / "periodograma_m1_2025.csv")
saida["periodograma_m1_2025"] = {"periodo": [round(v, 2) for v in pg["período (dias)"]], "ordenada": [round(v, 4) for v in pg["ordenada escalada"]],
                                 "g_alfa": round(pg["g_α (5%)"].iloc[0], 4), "pico": int(pg.index[pg["pico"]][0])}
fz = pd.read_csv(E / "sazonalidade_fisher_2025.csv", parse_dates=["início", "fim"])
def _f(serie, alvo, inicio, fim):
    r = fz[(fz["série"] == serie) & (fz["alvo"] == alvo) & (fz["início"] == pd.Timestamp(inicio)) & (fz["fim"] == pd.Timestamp(fim))].iloc[0]
    return {"p": r["p-valor"], "g": round(r["g"], 3), "g_alfa": round(r["g_α (5%)"], 3), "pico": r["pico da 1ª etapa (dias)"],
            "fundamental": int(r["período fundamental"]), "dias": int(r["dias"]), "achou_7": bool(r["período 7 significativo"])}
saida["fisher_2025"] = {"m1_fev": _f("Máquina 1", "demanda", "2025-02-01", "2025-03-23"), "total_fev": _f("Total", "demanda", "2025-02-01", "2025-03-23"),
                        "m1_quebra": _f("Máquina 1", "demanda", "2025-01-30", "2025-03-23"), "m1_jan": _f("Máquina 1", "demanda", "2025-01-01", "2025-03-23"),
                        "m1_mar": _f("Máquina 1", "demanda", "2025-03-01", "2025-03-23"), "m1_fev_treino": _f("Máquina 1", "demanda", "2025-02-01", "2025-03-09"),
                        "m2_fev": _f("Máquina 2", "demanda", "2025-02-08", "2025-03-23")}
pf = pd.read_csv(E / "perfil_fisher_m1_2025.csv", index_col=0).iloc[:, 0]
saida["perfil_fisher_m1_2025"] = {d: round(v, 1) for d, v in pf.items()}
jv = pd.read_csv(R / "structural_breaks/quebra_janela_vendas.csv", index_col=0)["valor"]
its = pd.read_csv(R / "structural_breaks/canibalizacao_horario_its.csv").set_index("janela").loc["±8 semanas"]
saida["horario"] = {"data": str(jv["data refinada (série inteira)"])[:10], "data_treino": str(jv["data detectada (só treino)"])[:10],
                    "fora_antes": round(float(jv["% fora 10–19h antes"]), 0), "fora_depois": round(float(jv["% fora 10–19h depois"]), 0),
                    "irr_total": round(float(its["IRR total"]), 2), "p_total": round(float(its["p total"]), 3),
                    "irr_nucleo": round(float(its["IRR nucleo"]), 2), "p_nucleo": round(float(its["p nucleo"]), 2)}
tv = pd.read_csv(R / "feature_analysis/testes_vazamento.csv").set_index("série")
imp = pd.read_csv(R / "feature_analysis/importancia_permutacao.csv")
janelas = imp["grupo"].str.startswith("janelas")
cd = pd.read_csv(R / "feature_analysis/contribuicao_dummies.csv")
irr = cd[(cd["modelo"] == "+ as duas dummies obrigatórias") & (cd["série"] == "Máquina 1")]["IRR das dummies"].iloc[0]
daic = cd[cd["modelo"] == "+ as duas dummies obrigatórias"].set_index("série")["ΔAIC vs só dia da semana"]
esp_total = json.loads((R.parent / "models/total_demanda_metadata.json").read_text())["especificação"]
saida["atributos"] = {"n": {s: int(tv.loc[s, "atributos"]) for s in tv.index},
                      "testes_ok": bool(tv.filter(like="teste").all().all()),
                      "jan_min": round(imp.loc[janelas, "aumento relativo %"].min(), 0), "jan_max": round(imp.loc[janelas, "aumento relativo %"].max(), 0),
                      "outros_max": round(imp.loc[~janelas, "aumento relativo %"].max(), 1),
                      "irr_horario": float(re.search(r"dummy_horario_reduzido = ([0-9.]+)", irr).group(1)),
                      "irr_novos": float(re.search(r"dummy_novos_clientes = ([0-9.]+)", irr).group(1)),
                      "daic_m1": round(daic["Máquina 1"], 0), "daic_total": round(daic["Total"], 0),
                      "exog_total": re.search(r"\+ \[(.*?)\]", esp_total).group(1)}
q = pd.read_csv(R / "model_comparison/qualidade_modelos_escolhidos.csv")
saida["qualidade"] = registros(q, 4)
ordem_grupos = ["janelas móveis (média/DP/CV)", "defasagens (lag_k)", "calendário e sazonalidade", "dummies de quebra e regimes",
                "comportamentais", "tendência (tempo)"]
impd = imp[imp["alvo"] == "demanda"].pivot(index="grupo", columns="série", values="aumento relativo %").reindex(ordem_grupos)
saida["importancia"] = {"grupos": ["Médias móveis (7/14/30 dias)", "Defasagens (1–28 dias)", "Calendário", "Dummies de quebra",
                                   "Comportamentais", "Tendência"],
                        "m1": [round(v, 1) for v in impd["Máquina 1"]], "total": [round(v, 1) for v in impd["Total"]],
                        "receita_jan": {s: round(v, 0) for s, v in
                                        imp[(imp["alvo"] == "receita") & janelas].set_index("série")["aumento relativo %"].items()}}
ORDEM_MODELOS = ["Naive sazonal", "Média 4 semanas", "Holt-Winters", "SARIMAX", "Poisson/NB2", "Random Forest"]
bt1 = bt[bt["série"] == "Máquina 1"].pivot(index="modelo", columns="alvo", values="MASE médio").reindex(ORDEM_MODELOS)
m2v = tvt[tvt["série"] == "Máquina 2"].set_index(["alvo", "modelo"])[["validação: MAE", "teste: MAE"]]
rel = {a: m2v.loc[a].reindex(ORDEM_MODELOS) / m2v.loc[a].loc["Naive sazonal"] for a in ("demanda", "receita")}
tot = bt[(bt["série"] == "Total") & (bt["alvo"] == "demanda")]
saida["evidencia"] = {"modelos": ["Naive", "Média 4 sem.", "Holt-Winters", "SARIMAX", "Poisson/NB2", "Random Forest"],
                      "m1_dem": [round(v, 3) for v in bt1["demanda"]], "m1_rec": [round(v, 3) for v in bt1["receita"]],
                      "m2_val": [round(v, 3) for v in rel["demanda"]["validação: MAE"]], "m2_teste": [round(v, 3) for v in rel["demanda"]["teste: MAE"]],
                      "m2_teste_rec": [round(v, 3) for v in rel["receita"]["teste: MAE"]],
                      "total_signif": int((tot["p Holm vs naive"] < 0.05).sum()), "total_p_max": round(tot.loc[tot["p Holm vs naive"] < 0.05, "p Holm vs naive"].max(), 3)}

SAIDA.write_text(json.dumps(saida, indent=1, ensure_ascii=False))
print(f"gravado: {SAIDA} ({len(saida)} blocos de dados, {len(saida['imagens'])} figuras)")
