# Entendimento do negócio

## 1. Problema de negócio

Uma operação de venda de café registra transações em dois pontos de venda: a Máquina 1 (`index_1.csv`, desde 01/03/2024) e a Máquina 2 (`index_2.csv`, desde 08/02/2025). O problema escolhido é **prever a demanda (transações/dia) e o faturamento (receita/dia) dos próximos 14 dias**. A previsão é feita para cada máquina e para a operação consolidada, e precisa respeitar as particularidades de cada uma: histórico, cardápio, preços, horário e público.

A demanda é baixa e ruidosa: de 5 a 20 vendas por dia, menos de 1 venda por hora. Ela também mudou de patamar várias vezes no período: houve extensão de horário, mudanças de preço, uma troca quase completa de clientela e a entrada da segunda máquina. Por isso, o problema não é só "ajustar um modelo". É **separar sinal de ruído e mudança estrutural de flutuação**, e dizer com honestidade quanto da demanda é previsível.

## 2. Oportunidade e aplicação das previsões

| Decisão operacional | Como a previsão ajuda | Horizonte |
|---|---|---|
| Reposição de insumos (café, leite, chocolate, copos) | volume esperado por máquina e dia da semana, com intervalo para estoque de segurança | 1–14 dias |
| Roteiro e frequência de abastecimento/limpeza | quais dias concentram vendas em cada máquina (a M2 vende 41% no fim de semana; a M1, 17%) | semanal |
| Planejamento de caixa | faturamento esperado por máquina e total, com faixa de incerteza | 14 dias |
| Monitoramento | queda ou alta fora do intervalo de previsão sinaliza falha, fechamento ou mudança de público | diário |

Os dados **não trazem** custos, margens, estoque, preço de insumos nem custo de ruptura. Por isso, este projeto **não estima ganho financeiro**. Os benefícios são descritos de forma qualitativa e pelas métricas de erro (ver `model_card.md`).

## 3. Objetivos analíticos

1. Verificar se `index_1` e `index_2` são máquinas distintas, o que condiciona somar as duas no Total.
2. Descrever a demanda e o faturamento: distribuição, sazonalidade semanal e horária, produtos e preços.
3. Tratar com critério ausentes, zeros e outliers, investigando cada caso sem remover nada automaticamente.
4. Identificar e testar as quebras estruturais, além de investigar duas hipóteses de negócio:
   - a quebra final (2025) está associada à entrada de **novos clientes**?
   - a mudança de horário (2024) provocou **canibalização**?
5. Construir atributos sem vazamento, incluindo as dummies das duas quebras.
6. Comparar 2 baselines e 4 modelos candidatos em 6 séries (3 séries × 2 alvos), com validação temporal, backtest e diagnóstico de resíduos.
7. Comparar o Total previsto diretamente com a soma das previsões das máquinas.
8. Propor uma arquitetura de produção com monitoramento de drift e critérios de retreino.

## 4. KPIs

**Técnicos**

| KPI | Definição | Por que |
|---|---|---|
| MAE | erro absoluto médio (transações/dia ou u.m./dia) | interpretável na unidade da operação |
| WAPE | Σ\|erro\| / Σ observado | "% do volume" errado; robusto a dias com zero |
| RMSE | raiz do erro quadrático médio | penaliza erros grandes (rupturas) |
| MAPE | média de \|erro\|/observado | só quando não há zeros no período |
| Viés (ME) | média(previsto − observado) | sub ou superestimação sistemática, que gera ruptura ou sobra |
| MASE | MAE ÷ MAE do naive sazonal | compara séries de escalas diferentes |
| Cobertura dos intervalos de 80/95% | % de dias dentro do intervalo | o intervalo é usado como estoque de segurança |
| Estabilidade | posto e MASE em 11 origens de 14 dias | um modelo bom numa só janela pode ser sorte |

**De negócio (para acompanhar em produção)**

- Erro diário dentro do intervalo de 80% em pelo menos ~80% dos dias.
- Viés acumulado em 4 semanas próximo de zero.
- Nº de alertas de anomalia (dias fora do intervalo de 95%) por máquina e mês.
- Os KPIs de ruptura e de desperdício só podem ser medidos depois de integrar dados de estoque, que hoje não existem.

## 5. Hipóteses investigadas

| # | Hipótese | Onde | Resultado (resumo) |
|---|---|---|---|
| H1 | `index_2` é outra máquina, e não uma cópia de `index_1` | `03_structural_breaks`, seção 1 | **Sustentada**: 13 de 13 testes favorecem "outra máquina". Há 0 de 262 vendas casadas, e o controle positivo detecta uma cópia de 5% |
| H2 | A demanda tem quebras estruturais de nível | seção 2 | **Sustentada**: 3 quebras na M1 (24/07/2024, 16/11/2024, 30/01/2025) e mudança de janela em 14–18/05/2024; sup-Wald com p < 0,001 |
| H3 | O aumento de patamar de 2025 veio de novos clientes | seção 3 | **Sustentada na M1**: 177 dos 196 cartões ativos depois da quebra são novos, e os cartões antigos quase desaparecem |
| H4a | A extensão de horário de 2024 canibalizou o núcleo 10–19h | seção 4a | **Não sustentada como dominante**: incrementalidade de 0,84 (IC 0,34–1,23). Uma canibalização parcial não é excluída |
| H4b | A entrada da M2 canibalizou a M1 | seção 4b | **Sem queda líquida da M1**. Há indício localizado no horário noturno, confundido com a troca de público |

## 6. Limitações das informações disponíveis

- **Não há identificador de máquina nem de local.** "Máquina 2 = `index_2`" é uma inferência, embora bem testada.
- **Não há horário de funcionamento registrado.** Ele é inferido pela primeira e última venda do dia, uma medida censurada.
- **O cliente é aproximado pelo cartão.** Um cartão não é uma pessoa; vendas em dinheiro não têm ID; a M2 não tem a coluna `card`.
- **O histórico é curto.** São 1 ano na M1 (nenhum ciclo anual completo para sazonalidade anual) e **44 dias na M2** (30 de treino).
- **Não há variáveis externas**: clima, eventos, feriados locais (o país não é informado), promoções, estoque ou falhas da máquina.
- **Não há custos nem margens.** Nenhum impacto financeiro pode ser calculado sem inventar números.
- **A moeda não é informada.** Valores em "unidades monetárias (u.m.)".
