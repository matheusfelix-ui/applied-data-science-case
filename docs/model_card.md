# Model card: previsão diária de demanda e faturamento (Coffee Sales)

## 1. Visão geral

| Item | Descrição |
|---|---|
| Tarefa | prever, para os próximos **14 dias**, a demanda (transações/dia) e o faturamento (u.m./dia) |
| Séries | Máquina 1 (`index_1`), Máquina 2 (`index_2`) e Total (modelado diretamente) |
| Saídas | previsão pontual diária + intervalos de 80% e 95% |
| Treino final | até 23/03/2025; previsão de 24/03 a 06/04/2025 |
| Artefatos | `models/<série>_<alvo>.joblib` + `_metadata.json`; `models/registro_modelos.csv` |
| Código | `src/models/`, `src/validation/`, `src/pipeline.py` |

## 2. Modelos em uso

| Série | Alvo | Modelo | Especificação na origem de teste (09/03) | Especificação da previsão final (reestimada até 23/03) |
|---|---|---|---|---|
| Máquina 1 | demanda | Holt-Winters | ETS(A,N,M) em √(y + 3/8), 8 semanas | ETS(A,Ad,M) em √(y + 3/8), 8 semanas |
| Máquina 1 | receita | Holt-Winters | ETS(A,N,A), 8 semanas | ETS(A,N,M), 8 semanas |
| Máquina 2 | demanda | Random Forest | profundidade 8, folha ≥ 5, 50% dos atributos, sem pesos de recência (regra de 1 EP) | idem |
| Máquina 2 | receita | Random Forest | idem | idem |
| Total | demanda | SARIMAX | SARIMA(0,1,1)(1,0,0)₇ + `maquinas_ativas`, `dummy_novos_clientes`, calendário; 8 semanas | SARIMA(0,1,1)(0,0,1)₇ + as mesmas exógenas, em √(y + 3/8); 8 semanas |
| Total | receita | SARIMAX | SARIMA(0,1,1)(1,0,1)₇ + as mesmas exógenas; 8 semanas | SARIMA(0,1,1)(1,0,1)₇ + exógenas e `feriado`; 16 semanas |

A **família** de cada série é fixada pela seleção. A **especificação interna** (janela, transformação, ordens) é escolhida de novo pela validação interna quando o modelo é reestimado com os dados mais recentes, exatamente como ocorrerá em produção a cada retreino. Os detalhes estão em `models/*_metadata.json`.

## 3. Desempenho

**Teste (10–23/03/2025) e backtest (11 blocos de 14 dias, 21/10/2024–23/03/2025):**

| Série · alvo | WAPE teste | MAE teste vs naive | MASE backtest | p (Holm) vs naive | Viés no teste |
|---|---|---|---|---|---|
| M1 · demanda | 22,4% | −7% | **0,77** | 0,07 | +8% |
| M1 · receita | 20,4% | −10% | 0,82 | 0,32 | −4% |
| M2 · demanda | 41,1% | −19% | — | — | +8% |
| M2 · receita | 37,3% | −23% | — | — | +6% |
| Total · demanda | 20,6% | −20% | 0,89 | **0,001** | −6% |
| Total · receita | 19,5% | −24% | 0,92 | **0,005** | −1% |

**Referência de ruído irredutível (piso de Poisson, WAPE):** vendas: ~21% (M1), ~32% (M2) e ~17% (Total); receita: ~21%, ~33% e ~18%. Medido nos últimos 28 dias do treino. Os erros dos modelos escolhidos ficam próximos desses pisos.

**Intervalos:** a cobertura do intervalo de 80% no teste fica entre 64% e 100% entre os modelos (média de 89%). Com 14 dias, a cobertura é uma estimativa ruidosa.

**Total direto × soma das máquinas (teste):**
- na demanda, o MAE é 4,13 no direto e 3,59 na soma;
- na receita, 117 no direto e 107 na soma.

A soma foi melhor nesta janela. A M2 tem pouco histórico para que o modelo direto aprenda a sua entrada. Ver a recomendação na seção 6.

**Ressalva:** o teste entra na seleção (peso de 40%), então o erro de teste do modelo escolhido é otimista. A melhor estimativa do desempenho esperado é o **MASE do backtest**.

## 4. Uso pretendido e fora do escopo

**Pretendido:**
- planejamento de reposição de insumos (com o limite superior do intervalo como estoque de segurança);
- roteiro de abastecimento;
- planejamento de caixa;
- detecção de anomalias (dias fora do intervalo de 95%).

**Fora do escopo:**
- previsão horária (volume insuficiente);
- previsão de longo prazo (> 14 dias);
- máquinas novas sem pelo menos ~4 semanas de histórico;
- decisões financeiras automáticas (não há custos nem margens);
- inferência causal (os efeitos de horário e preço observados são associações).

## 5. Dados

- **Fonte:** Kaggle `ihelon/coffee-sales` v21; 3.636 vendas (M1, 01/03/2024–23/03/2025) e 262 (M2, 08/02–23/03/2025).
- **Atributos:** calendário, defasagens, janelas móveis, dummies de quebra (detectadas só com o passado), `maquinas_ativas` e comportamentais congelados na origem. Os dois testes automáticos de vazamento passam.
- **Pré-processamento:** dias sem venda = 0, sem imputação; nenhum outlier removido; a M2 começa em 08/02/2025.

## 6. Resultados e impacto de negócio

- **Precisão.** O erro típico é de ~20% do volume diário na M1 e no Total e de ~40% na M2. Para uma máquina que vende ~15 vendas/dia, isso é um erro médio de ~3 vendas/dia, metade do qual é ruído de contagem irredutível.
- **Erros por máquina e período.** O erro sobe em todos os modelos nos blocos em torno da quebra de 2025: no Holt-Winters da M1, o MASE vai de 0,74 para 0,81; no SARIMAX, de 0,73 para 1,41. Fora das quebras, os modelos ficam ~25% abaixo do naive.
- **Mudanças de comportamento.**
  - Os modelos de nível adaptativo (Holt-Winters) absorvem um novo patamar em poucas semanas.
  - A detecção formal de uma quebra leva ~3–4 semanas.
  - Eventos conhecidos (como a entrada da M2) entram como variáveis (`maquinas_ativas`), que o SARIMAX do Total usa.
- **Aplicações operacionais.**
  - Reposição e roteiro por dia da semana: a M1 vende ~60% menos no fim de semana; a M2 concentra 41% das vendas no fim de semana.
  - Faturamento consolidado esperado de 8.771 u.m. (Total direto) a 9.514 u.m. (soma das máquinas) em 24/03–06/04/2025.
  - Alerta para dias zerados atípicos (os 7 zeros da M1 são implausíveis como falta de clientes).
- **Total oficial = soma das máquinas** (Holt-Winters M1 + Random Forest M2). A operação decide por máquina, o total fica coerente com as partes e, no teste, a soma errou 13% menos na demanda e 8% menos na receita que o SARIMAX direto. O SARIMAX direto segue em paralelo como alarme de coerência e plano B. A evidência é de uma única janela de 14 dias (a M2 só existe desde 08/02/2025): reavaliar com backtest quando a M2 tiver 3 meses ou mais. A evolução é a reconciliação hierárquica.
- **Benefícios potenciais (qualitativos).** Menos rupturas e sobras de insumos, roteiros mais aderentes à demanda, detecção precoce de falhas. **Nenhum valor financeiro é estimado**, porque os dados não trazem custos, margens nem estoques.

## 7. Limitações e riscos

As principais estão abaixo; o detalhamento está em `docs/limitations.md`.
- A M2 tem 30 dias de treino e não foi avaliada em backtest.
- Há 1 ano de histórico: não há sazonalidade anual.
- O regime vigente é suposto persistente no horizonte, e uma mudança operacional nova degrada as previsões até ser detectada.
- O Holt-Winters tem oscilação numérica de ~0,03 transação/dia entre execuções.

## 8. Manutenção

O ciclo está descrito em `docs/architecture.md`:
- retreino semanal, com seleção completa mensal;
- gatilhos: alerta de performance por 7 dias ou nova quebra;
- monitoramento de WAPE, viés e cobertura em janelas de 28 dias, comparando sempre com a Média 4 semanas.
