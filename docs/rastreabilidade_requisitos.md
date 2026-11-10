# Rastreabilidade dos requisitos

Cada item do enunciado do desafio e onde ele está atendido no repositório. As seções seguem a numeração do enunciado. A coluna "Observação" registra decisões que se afastam da letra do enunciado e o motivo.

## 1–2. Objetivo e estrutura

| Requisito | Onde está | Observação |
|---|---|---|
| Desenvolver em pasta nova, preservando os originais | `case_coffee_sales/`; os notebooks de origem ficam fora do repositório e não foram alterados | README, "Origem do trabalho" |
| Estrutura de diretórios pedida | todos os arquivos e pastas da árvore do enunciado existem | Há módulos extras: `eda/duplicate_analysis.py`, `eda/trend_seasonality.py`, `validation/meta_decisao.py`. O notebook 05 (tendência e sazonalidade) foi acrescentado, então modelagem e validação ficaram como 06 e 07. |

## 3. Entendimento do negócio

| Requisito | Onde está |
|---|---|
| Problema, oportunidade, objetivos analíticos | `docs/business_understanding.md`, seções 1–3 |
| KPIs técnicos e de negócio | `docs/business_understanding.md`, seção 4 |
| Hipóteses investigadas | `docs/business_understanding.md`, seção 5 |
| Limitações das informações | `docs/business_understanding.md`, seção 6 |
| Sem custos, margens ou ganhos inventados | todas as recomendações são pilotos medidos contra a previsão (`docs/model_card.md`, seção 6; apresentação, slides 3, 13 e 15) |

## 4. Identificação das máquinas

| Requisito | Onde está |
|---|---|
| Reaproveitar o código existente, preservando a metodologia | `src/structural_breaks/machine_identification.py`: porte de `teste_index2_nova_maquina.ipynb`, com as mesmas sementes e ordem do gerador aleatório |
| Resultados e evidências | `reports/structural_breaks/identificacao_maquinas/` (`sintese_testes.csv`, `controle_positivo.csv`, `correspondencia.csv`, `perfis_periodo_comum.png`); notebook 01; apresentação, slide 4 |

## 5. Análise exploratória

| Requisito | Onde está |
|---|---|
| Distribuição de vendas e faturamento; evolução temporal | `reports/eda/01_distribuicoes.png`, `02_evolucao.png`, `05_mensal.png` |
| Horário e dia da semana | `reports/eda/03_hora_dia_semana.png`, `04_perfis.png`, `13_perfil_semanal_m1.png` |
| Análise individual e consolidada | M1, M2 e Total em todas as figuras de evolução; notebook 02 |
| Produtos e mix | `reports/eda/09_mix_produtos.png`, `10_evolucao_mix_precos.png`, `11_ticket_medio.png`, `12_pareto_produtos.png`; `src/eda/product_analysis.py` |
| Valores ausentes: quantificação, padrões, impacto | `src/eda/missing_values.py`; `reports/eda/06_ausentes.png`; apêndice A1 da apresentação |
| Zeros: significado, zero × ausente, tratamento | `src/eda/zero_analysis.py`; `reports/eda/07_zeros.png`; `src/data/preprocessing.py` (semântica: dia sem registro = 0; M2 antes de 08/02/2025 = fora da série) |
| Outliers: detecção, causas, legítimo × inconsistência, impacto | `src/eda/outlier_analysis.py`; `reports/eda/08_outliers.png`, `outliers_calendario.csv` (6 de 8 quedas em feriado ou fim de semana); nada é removido |
| Não remover extremos nem zeros automaticamente | `docs/methodology.md`, seção 2 |

## 6. Quebras estruturais e hipóteses

| Requisito | Onde está | Observação |
|---|---|---|
| 6.1 Pontos de mudança, testes, estatísticas e p-valores | `src/structural_breaks/structural_break_test.py`: Binseg + BIC, sup-Wald de Andrews (HAC), OLS-CUSUM, regressão segmentada. Resultados em `reports/structural_breaks/testes_globais.csv`, `regressao_segmentada.csv`, `confirmacao_local_*.csv`, `01_quebras_nivel.png`. Tendência × nível em `src/eda/trend_seasonality.py` e no notebook 05. | — |
| 6.2 Novos clientes: novos × recorrentes, frequência, únicos, contribuição | `src/structural_breaks/customer_analysis.py`; `reports/structural_breaks/clientes_*.csv` e `03_novos_clientes.png` | O cartão é o proxy de cliente; compras em dinheiro e a M2 (sem coluna `card`) ficam fora da identificação |
| 6.3 Canibalização pela mudança de horário | `src/structural_breaks/cannibalization_analysis.py`; `reports/structural_breaks/canibalizacao_*.csv`, `04_canibalizacao_horario.png`, `05_canibalizacao_entrada_m2.png` | A mudança de horário (mai/2024) aconteceu quando só a M1 existia: a M2 começa em 08/02/2025. Por isso a canibalização *entre máquinas* pelo horário não é testável. A análise foi feita em duas partes, como combinado: (a) faixas horárias da M1 antes × depois da ampliação; (b) efeito da entrada da M2 sobre a M1, com participação relativa de cada máquina. Nenhuma hipótese operacional alheia à canibalização foi incluída. |

## 7. Atributos

| Requisito | Onde está | Observação |
|---|---|---|
| Temporais, defasagens, janelas móveis, comportamentais | `src/features/temporal_features.py`, `lag_features.py`, `feature_pipeline.py`; notebook 04; apêndice A5 | A hora entra como atributo comportamental (proporção de vendas de manhã e à noite), porque a série é diária (`docs/methodology.md`, seção 4) |
| Dummy da quebra final (novos clientes) | `dummy_novos_clientes` em `src/features/break_dummies.py`: 30/01/2025 na M1, 04/02/2025 no Total | Data fundamentada em `reports/structural_breaks/datas_dummies.csv` |
| Dummy da quebra inicial (horário reduzido) | `dummy_horario_reduzido` = 1 antes da ampliação: 18/05/2024 só com o treino, refinada para 14/05/2024 | Idem; `quebra_janela_vendas.csv` |
| Sem informação futura | atributo do dia d usa só dados até d − 1; testes `test_sem_vazamento_do_futuro` e `testes_vazamento.csv`; quebras redetectadas em cada origem | — |

## 8. Modelagem

| Requisito | Onde está |
|---|---|
| 2 baselines e 4 candidatos | `config.yaml` (`modelos`): Naive sazonal, Média 4 semanas; Holt-Winters, SARIMAX, Poisson/NB2, Random Forest |
| Justificativa, alternativas, trade-offs, descartes | `docs/methodology.md`, seções 5.1–5.3; apêndice A4 |
| Modelos independentes para M1, M2 e Total | `models/machine_1_*`, `machine_2_*`, `total_*` (joblib + metadados) |
| Total modelado diretamente, mais a soma para comparação | `docs/methodology.md`, seção 8; `reports/forecasts/total/total_direto_vs_soma_maquinas.csv`; `reports/model_comparison/04_total_direto_vs_soma.png` |

## 9. Validação

| Requisito | Onde está |
|---|---|
| Treino/teste cronológico, sem embaralhar | `src/validation/backtesting.py` (`datas_divisao`): teste = últimos 14 dias |
| Backtesting com janelas | 11 origens de 14 dias sem sobreposição, com tudo refeito em cada origem; `reports/model_comparison/backtest_*.csv`, `02_backtest_mase.png`, `03_backtest_origens_*.png` |
| Generalização, resíduos, desempenho por período | `treino_validacao_teste.csv` e `05_treino_validacao_teste.png`; `diagnostico_residuos.csv` e `residuos/` (6 séries); `backtest_erro_por_periodo.csv` |
| MAE, RMSE, WAPE, MAPE, viés | `src/validation/metrics.py`; `reports/model_comparison/metricas_teste.csv`, `qualidade_modelos_escolhidos.csv` (também R², MASE e cobertura dos intervalos) |
| Resultados de treino, validação e teste | `reports/model_comparison/treino_validacao_teste.csv` |
| Escolha com peso importante para o teste, estabilidade e negócio | `config.yaml` (`selecao`): 40% posto no teste, 35% backtest, 25% estabilidade e filtro de viés ≤ 15%; `selecao_final.csv` |
| Só 2 baselines e 4 candidatos apresentados | os demais modelos testados aparecem apenas como descartes justificados (`docs/methodology.md`, seção 5.2) |

## 10. Previsões finais

| Requisito | Onde está |
|---|---|
| 6 combinações (M1, M2, Total × demanda, faturamento) | `reports/forecasts/{machine_1,machine_2,total}/previsao_futura_14_dias.csv`, com intervalos de 80% e 95% |
| Total independente × soma | `reports/forecasts/total/total_direto_vs_soma_maquinas.csv` e `.png` |
| Previsões, métricas e gráficos separados por série | uma pasta por série em `reports/forecasts/` (`metricas_teste.csv`, `previsoes_modelo_escolhido.png`) |

## 11. Visualizações

| Requisito | Onde está |
|---|---|
| Matplotlib/Seaborn, paleta padronizada, títulos, eixos, grades discretas, anotações de quebras | `src/visualization/plots.py`; paleta em `config.yaml` (`visualizacao`) |
| PNG em alta resolução + PDF | 300 dpi; 44 figuras, cada uma em PNG e PDF, em `reports/` |
| EDA, ausentes, zeros, outliers, produtos, quebras, canibalização, atributos, comparação de modelos, resíduos, previsões | `reports/eda/` (15), `structural_breaks/` (7), `feature_analysis/` (3), `model_comparison/` (14, incluindo resíduos), `forecasts/` (5) |

## 12. Engenharia de software

README, seção 6 (tabela de evidências) e seção 5 (reprodutibilidade, com a verificação num clone limpo).

## 13. Arquitetura para produção

`docs/architecture.md`, seções 1–7: entrada, processamento, atributos, treino e armazenamento, disponibilização, monitoramento, drift, retreino, versionamento, escalabilidade e manutenção. O documento distingue o que já está implementado do que é proposta. Resumo na apresentação, slide 14; detalhe no apêndice A9.

## 14. Resultados e impacto de negócio

`docs/model_card.md`, seções 3 e 6, e a apresentação:
- **slides 11–12:** precisão por máquina, previsões e incerteza;
- **slides 6 e 9:** mudanças de comportamento capturadas;
- **slide 13:** aplicação no planejamento.

Nenhum impacto financeiro é estimado.

## 15. Evoluções futuras

`docs/limitations.md`, seções 1–6: limitações, riscos, dados adicionais, melhorias metodológicas, evoluções da arquitetura e próximos passos. Resumo na apresentação, slide 15.

## 16. Apresentação executiva

`reports/apresentacao/coffee_sales_executiva.pptx` tem 16 slides principais (~27 min pelas notas) e apêndice técnico navegável (A1–A12):

| Seção pedida | Slides |
|---|---|
| 1. Contexto e oportunidade | 2–3 |
| 2. Dados e EDA | 3 (definição da série), 4, 5, 7, 8 |
| 3. Hipóteses e quebras estruturais | 6, 9 |
| 4. Alternativas e decisões metodológicas | 10, 11 |
| 5. Atributos e desenvolvimento dos modelos | 10, 11 |
| 6. Resultados técnicos e impacto de negócio | 11, 12, 13, 16 |
| 7. Arquitetura para produção | 14 |
| 8. Limitações e evoluções futuras | 15 |

## 17–18. Entregáveis e execução

| Entregável | Onde está |
|---|---|
| Projeto Python organizado e documentado | `src/` (docstrings em 220 de 251 funções públicas) |
| Notebooks de apoio | `notebooks/01`–`07`, executados e sem erros |
| Testes automatizados | `tests/`: 82 testes, também rodados no CI |
| `requirements.txt` | versões exatas |
| README com instruções | `README.md`, seção 4 |
| Gráficos e relatórios | `reports/` |
| Modelos treinados | `models/` |
| Previsões finais das três séries | `reports/forecasts/` |
| Metodologia e arquitetura | `docs/` |
| Apresentação | `reports/apresentacao/coffee_sales_executiva.pptx` |
