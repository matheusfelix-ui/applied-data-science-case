# Metodologia

Este documento justifica as escolhas metodológicas. Os números citados vêm de `reports/` e podem ser reproduzidos com `python -m src.pipeline`.

## 1. Fluxo

```
dados brutos ─► validação ─► séries diárias ─► EDA ─► quebras e hipóteses ─► atributos ─► 6 modelos × 6 séries
     │                                                                                      │
     └── identificação das máquinas (index_2 ≠ index_1)                  validação interna ─┤
                                                                         teste (14 dias)  ──┤
                                                                         backtest (11 origens)
                                                                                            ▼
                                                              seleção (teste + backtest + estabilidade + viés)
                                                                                            ▼
                                                              previsões finais (14 dias) + modelos versionados
```

## 2. Decisões sobre os dados

| Decisão | Justificativa | Evidência |
|---|---|---|
| `index_1` = Máquina 1, `index_2` = Máquina 2; Total = soma | 13 de 13 testes rejeitam "index_2 repete index_1"; o controle positivo detecta cópias de 5% | `reports/structural_breaks/identificacao_maquinas/` |
| Granularidade **diária** | < 1 venda por hora; uma série horária seria dominada por ruído de contagem | heatmap hora × dia (`reports/eda/03_*`) |
| Zeros mantidos como zero | zero vendas ⇒ zero receita; imputar inventaria vendas; 7 de 388 dias | `reports/eda/dias_zerados.csv` (P(Y=0) explica cada zero) |
| Nenhum outlier removido | atípicos são legítimos ou fazem parte de quebras; winsorizar não muda o erro (Δ MAE ≈ 0) | `reports/eda/outliers_*.csv` |
| M2 começa em 08/02/2025 (sem zeros antes) | antes disso a máquina não existe | `preprocessing.py` |
| Pedidos com vários itens mantidos | 7 grupos de horários idênticos na M2 com produtos diferentes = PDV | notebook 01 |

## 3. Quebras estruturais

- **Detecção sem vazamento:** Segmentação Binária + BIC, refeita **só com os dados até cada origem**. A verificação com tamanho efetivo de amostra escolhe as mesmas datas.
- **Testes formais:**
  - **sup-Wald de Andrews com HAC**, porque com data desconhecida o Chow comum dá p inválido;
  - OLS-CUSUM;
  - LR das dummies em NB2;
  - regressão segmentada para separar nível de tendência;
  - sup-Wald local para confirmar cada quebra (lógica de Bai-Perron).
- **Resultado:**
  - quebras de **nível** (não de tendência) na M1 em 24/07/2024, 16/11/2024 e 30/01/2025 (Total: 04/02/2025);
  - mudança da janela de vendas em 14–18/05/2024.

  Todas têm p < 0,001.

### Dummies obrigatórias

| Dummy | Data (modelagem) | Fundamentação |
|---|---|---|
| `dummy_horario_reduzido` (1 antes da extensão) | 18/05/2024 (refinada: 14/05) | % de vendas fora de 10–19h: 12% → 40%; maior quebra única da demanda da M1 (sup-Wald) |
| `dummy_novos_clientes` (1 a partir do novo patamar) | 30/01/2025 (M1), 04/02/2025 (Total) | sup-Wald local p < 0,001; 177 de 196 cartões ativos depois são novos |

As duas reduzem o AIC de uma NB2 com dia da semana em 56 (M1) e 113 (Total) pontos.

## 4. Atributos

Os atributos e as garantias de não vazamento estão descritos em `src/features/feature_pipeline.py` e no notebook 04. Os pontos principais:
- todo atributo de um dia d usa só valores até d − 1;
- no horizonte, a previsão é **recursiva**;
- os comportamentais ficam congelados no valor da origem;
- dois testes automáticos (futuro → ruído; cauda × série inteira) rodam no CI.
- **hora**: a série é diária (há menos de 1 venda por hora), então a hora não indexa a série. Ela entra como atributo comportamental: proporção de vendas de manhã (até 09h59) e à noite (a partir de 20h), definida em `config.yaml`.

### 4.1 Tendência, sazonalidade e ciclo

O notebook 05 (`src/eda/trend_seasonality.py`) caracteriza a estrutura temporal da M1 e do Total, usando só o treino até 09/03/2025.

| Diagnóstico | Resultado | Consequência |
|---|---|---|
| Tendência (Mann-Kendall com Hamed-Rao; linear × degraus) | significativa só no ano inteiro; em 4 de 5 regimes, não; degraus vencem a reta por ΔBIC de 49 a 70 | sem tendência linear; ETS(·,N,·); dummies de regime; janelas curtas |
| Sazonalidade (periodograma + g de Fisher, como em `teste.ipynb`) | com tendência polinomial, picos espúrios de 62 a 187 dias, resíduos dos degraus; com degraus, nenhum pico significativo | o teste do maior pico precisa de uma tendência compatível com as quebras |
| Sazonalidade semanal (F do dia da semana por regime) | significativa só em 2025 (M1: R² = 81%; fim de semana −45%) | perfil estimado só no regime atual (8 semanas) |
| Ciclo (STL + degraus + HP; Fisher semanal; dia do mês) | desvio persistente, mas sem período e sem efeito de dia do mês | nada determinístico a modelar |
| ACF/PACF e estacionariedade | estacionária em torno dos degraus; Δ1 deixa um MA(1) com θ ≈ −1; Δ7 é excessiva | SARIMA(0,1,1)(P,0,Q)₇, ou d = 0 com dummies; D = 0 |

## 5. Modelos

### 5.1 Escolha dos candidatos

A escolha partiu da evidência acumulada em `modelagens_testes.ipynb`, que comparou 16 modelos e depois 8, além de NHITS/NBEATSx, em 11 origens. O critério foi cobrir as estruturas relevantes da série com o menor conjunto de modelos complementares:

| Característica da série | Como é tratada | Modelo |
|---|---|---|
| Sazonalidade semanal forte | componente sazonal explícito | Holt-Winters, SARIMAX |
| Mudanças de nível abruptas | janela de estimação como hiperparâmetro; dummies de quebra | todos os candidatos; dummies no SARIMAX, no modelo de contagem e na RF |
| Contagens pequenas (variância ∝ média) | distribuição de contagem; transformação de Anscombe | Poisson/NB2; HW e SARIMAX com Anscombe candidata |
| Relações não lineares e interações | árvores | Random Forest |
| Pouco histórico (M2) | modelos que degeneram para "nível + ruído"; regra de 1 erro-padrão | SARIMAX/Poisson; RF regularizada |

| Modelo | Família | Seleção interna | Vantagens | Limitações |
|---|---|---|---|---|
| **Naive sazonal** | baseline | — | referência mínima; captura o padrão semanal | carrega o ruído da última semana |
| **Média 4 semanas** | baseline | — | estável; adversário exigente | atrasa nas mudanças de nível |
| **Holt-Winters (ETS)** | estatístico | janela × transformação (MAE de validação); ETS por AICc | adapta o nível rapidamente; intervalos analíticos; barato | não usa exógenas; oscilação numérica pequena |
| **SARIMAX** | estatístico | janela × transformação; ordens por AICc; exógenas por AICc | incorpora dummies e `maquinas_ativas`; inferência sobre os coeficientes | instável perto das quebras na M1 (MASE 1,41 depois de fev/2025) |
| **Poisson/NB2** | estatístico (contagem) | janela (MAE); família × especificação (AIC) | intervalos coerentes para contagens; receita por Poisson composto | supõe que o regime vigente persiste |
| **Random Forest** | ML | 2 estágios + regra de 1 EP (WAPE de validação, recursivo) | não linearidades; robusta a hiperparâmetros | não extrapola; precisa de histórico; mais cara |

### 5.2 Alternativas consideradas e descartadas

Os números vêm de `modelagens_testes.ipynb`.

| Alternativa | Motivo do descarte |
|---|---|
| Croston / SBA / TSB | a demanda **não é intermitente** (ADI 1,02–1,07; CV² 0,28–0,38 → classe "suave") |
| GARCH / EGARCH / SARIMA-GARCH | ARCH-LM não rejeita (p 0,37–0,92): a heterocedasticidade é de **nível**, tratada pelo modelo de contagem |
| TBATS | descartou os harmônicos em 5 de 6 séries; equivale a ETS(A,N,N), mais caro e pior |
| Prophet | feito para vários ciclos anuais e tendências suaves; ficou entre os 3 piores nas origens da M1 |
| Theta / ETS só por AICc / DHR | redundantes com ETS ou SARIMA |
| XGBoost, LightGBM, CatBoost | na metade de baixo em 3 de 4 séries nas 11 origens; a RF foi mais estável, e LightGBM/XGBoost exigem libomp |
| Modelo conjunto (uma árvore para M1 + M2) | piora a M2 em 46% a 90% |
| SARIMA puro | o SARIMAX contém o SARIMA (exógenas por AICc, com ΔAICc reportado) |
| *Ensemble* SARIMA + XGBoost | bom nas origens (2º na M1), mas é uma combinação, menos interpretável e mais difícil de manter; combinações ficam como evolução futura |
| NHITS / NBEATSx (com Optuna) | metade de baixo em todas as séries; memorizam o ruído (WAPE in-sample de 2–4% contra um piso de 21–26%) |

### 5.3 Trade-offs

- **Precisão × estabilidade.** A RF venceu 5 de 6 séries numa única origem na versão anterior, mas esse domínio não se repetiu em 11 origens. O Holt-Winters com janela curta foi o mais consistente na M1.
- **Interpretabilidade × flexibilidade.** SARIMAX e contagem dão coeficientes interpretáveis (ex.: IRR das dummies). A RF dá apenas importâncias.
- **Custo.** Os estatísticos treinam em menos de 1 a 10 s por série; a RF, ~30 s com seleção. Todos cabem num pipeline diário.

## 6. Validação

| Camada | Desenho | Uso |
|---|---|---|
| Treino | 01/03/2024 (M2: 08/02/2025) a 09/03/2025 | estimação; WAPE de ajuste comparado ao **piso de ruído** de Poisson |
| Validação interna | 3 dobras de 14 dias (origens 26/01, 09/02, 23/02); a M2 tem 1 | escolha de janela, transformação, especificação e hiperparâmetros |
| Teste | 10/03–23/03/2025 | avaliação final, nunca usado no ajuste |
| Backtest | 11 blocos de 14 dias sem sobreposição (21/10/2024–23/03/2025), tudo refeito em cada origem | estabilidade, significância (teste t do ΔMAE por origem, Holm) e erro por período |

- **Métricas:** MAE, RMSE, WAPE, MAPE (só sem zeros), sMAPE, ME (viés), MASE, cobertura dos intervalos de 80/95% e Winkler.
- **Resíduos:** Ljung-Box, Shapiro, Kruskal por dia da semana e viés dos erros de teste (`reports/model_comparison/diagnostico_residuos.csv`).
- **Piso de ruído.** Mesmo conhecendo o nível local exato, um modelo de média erraria, nos últimos 28 dias do treino, ~21% (M1), ~17% (Total) e ~32% (M2) de WAPE nas vendas (~21%, ~18% e ~33% na receita). Um ajuste in-sample **abaixo** do piso indica sobreajuste, não mérito.
- **Meta-decisão registrada.** A regra de seleção da RF (1 erro-padrão × mínimo) foi decidida no backtest de 10 origens **anteriores ao teste**. Houve empate (MASE médio 0,862 × 0,868), e a regra mais simples foi mantida (`reports/model_comparison/meta_decisao_regra_rf_backtest_pre_teste.csv`).

## 7. Seleção final

Os pesos estão em `config.yaml` (seção `selecao`):

```
escore = 0,40 · posto(MAE no teste) + 0,35 · posto(MASE médio no backtest) + 0,25 · posto(% de origens em que bate a Média 4 semanas)
```

- **Filtro de negócio:** |ME| / média do teste > 15% desclassifica o modelo.
- **Máquina 2** (sem histórico para backtest): o backtest é substituído pelo MAE de validação interna, e a estabilidade pelo |viés| no teste.
- **Ressalva:** como o teste entra na escolha, o erro de teste do modelo escolhido é otimista. O MASE do backtest é a estimativa de desempenho esperado.

## 8. Total: modelo independente × soma das máquinas

O Total é modelado **diretamente**, como uma série própria, com `maquinas_ativas` e as dummies. A soma das previsões de M1 e M2 do mesmo modelo é calculada à parte, para comparação (`reports/model_comparison/total_direto_vs_soma_maquinas.csv` e `reports/forecasts/total/`).

## 9. Reprodutibilidade

- **Sementes fixas:** 2024 nos modelos e 42 no bootstrap das análises causais (as mesmas do notebook original).
  - Isso inclui a simulação dos intervalos do ETS com sazonalidade multiplicativa (`rng` em `get_prediction`).
  - `test_holt_winters_reprodutivel` garante previsões e intervalos idênticos.
- **Versões fixadas:** `requirements.txt` e `reports/apresentacao/fonte/package-lock.json`.
- **Hash dos dados:**
  - `MANIFEST.json` é **conferido** a cada execução; arquivo divergente interrompe o pipeline.
  - Os metadados dos modelos guardam o hash dos dados e o da configuração.
- **Pipeline:** `python -m src.pipeline` regenera todos os relatórios em ~10–13 min.
- **Meta-decisão da Random Forest:** `python -m src.validation.meta_decisao` refaz a comparação da regra de seleção.
- **Verificação:** num clone limpo, com ambiente novo, 98 de 101 tabelas saíram idênticas byte a byte. As outras diferem só em colunas de relógio (`tempo_s`, `gerado_em`). Detalhes no README, seção 5.
