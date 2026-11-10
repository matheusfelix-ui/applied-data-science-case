# Limitações, riscos e evoluções futuras

## 1. Limitações atuais

**Dos dados**

| Limitação | Consequência | Onde aparece |
|---|---|---|
| A Máquina 2 tem 44 dias (30 de treino) | não há backtest da M2; a validação interna tem uma só dobra; a sazonalidade semanal é imprecisa | notebooks 06–07; a seleção da M2 usa teste + validação + viés |
| 1 ano de histórico na M1 | sazonalidade anual não identificável: um mês alto pode ser sazonal ou um regime | notebook 02 |
| Não há identificador de máquina ou local | "Máquina 2 = `index_2`" é inferida (13 testes), não registrada | notebook 01 |
| Horário de funcionamento não registrado | a janela é inferida pela 1ª e última venda (medida censurada); os zeros de fechamento não se distinguem de falta de clientes com certeza | notebooks 02–03 |
| Cliente aproximado pelo cartão | um cartão não é uma pessoa; sem ID em dinheiro; a M2 sem `card` | notebook 03, seção 3 |
| Sem variáveis externas (clima, eventos, feriados locais, promoções) | parte da variação diária é tratada como ruído | piso de ruído de ~17–32% de WAPE nas vendas (M1 21%, M2 32%, Total 17%) |
| Sem custos, margens, estoque ou rupturas | não é possível calcular impacto financeiro nem otimizar o estoque | `business_understanding.md` |
| Moeda e país não informados | valores em "u.m."; feriados genéricos | `config.yaml` |

**Metodológicas**

- **Teste curto.** São 14 dias por série: as métricas de teste têm muita variância, e diferenças de 10% entre modelos raramente são significativas (Diebold-Mariano). O backtest de 11 origens mitiga isso na M1 e no Total.
- **O teste entra na seleção final**, como pede o enunciado. Por isso o MAE de teste do modelo escolhido é otimista como estimativa do erro futuro. A referência honesta é o MASE médio do backtest.
- **Quebras detectadas com atraso.** Sem informação externa, a quebra de fev/2025 só é detectável ~3–4 semanas depois. Nesse intervalo os modelos subestimam o novo patamar.
- **O regime vigente persiste no horizonte.** Os modelos supõem que o regime da origem continua nos 14 dias seguintes. Uma nova quebra dentro do horizonte não é prevista.
- **As análises causais são observacionais.** A incrementalidade da extensão de horário e a (ausência de) canibalização pela M2 vêm de antes/depois sem grupo de controle.
- **Atributos comportamentais congelados.** No horizonte, eles ficam fixos no último valor conhecido. É uma aproximação, mas a validação mostrou que eles têm pouco valor preditivo.
- **Oscilação numérica do ETS.** O Holt-Winters tem variação de ~0,03 transação/dia entre execuções (otimizador). Isso não afeta conclusões, mas impede igualdade bit a bit.

## 2. Riscos identificados

| Risco | Probabilidade / impacto | Mitigação |
|---|---|---|
| Nova mudança operacional (horário, preço, local) muda o patamar | alta (aconteceu 4 vezes em 1 ano) / alto | cadastro de eventos operacionais como entrada; CUSUM nos resíduos; retreino por gatilho |
| A Máquina 2 se comportar diferente com mais histórico | alta / médio | intervalos largos; reavaliação mensal; incluir a M2 no backtest a partir de ~3 meses |
| Fechamento ou falha da máquina interpretado como queda de demanda | média / médio | alerta de anomalia (fora do IC 95%); cadastro de indisponibilidade |
| Mudança de preço altera a receita sem alterar a demanda | média / médio na receita | ticket médio móvel; monitorar o ticket separadamente da demanda |
| Mudança no formato do log (ex.: nova coluna, outro relógio) | baixa / alto | validação de contrato bloqueia a carga |
| Uso das previsões pontuais sem o intervalo | média / médio | publicar sempre os intervalos de 80/95%; dimensionar estoque pelo limite superior |

## 3. Dados adicionais que melhorariam a solução

1. **Cadastro de eventos operacionais:** horário de funcionamento, manutenção, indisponibilidade, mudança de local e de preço. Transforma quebras "descobertas" em variáveis conhecidas.
2. **Identificador de máquina e local** em cada transação, e a coluna `card` também na M2.
3. **Estoque, reposições e rupturas:** permitem medir demanda censurada (venda perdida por falta de produto) e calcular custo de erro.
4. **Custos e margens por produto:** permitem otimizar estoque e horário com critérios financeiros reais.
5. **Calendário local** (país e cidade): feriados e eventos.
6. **Clima** (temperatura e chuva), que afeta bebidas quentes e frias.
7. **Mais histórico** (2+ anos) para estimar sazonalidade anual.
8. **Mais máquinas** para modelos globais.

## 4. Melhorias metodológicas

- **Reconciliação hierárquica** (MinT) entre M1, M2 e Total, em vez de escolher entre o modelo direto e a soma.
- **Detecção online de quebras** com menor atraso (CUSUM/BOCPD sobre os resíduos), alimentando as dummies.
- **Combinação de previsões** (média dos 2–3 melhores modelos do backtest). Combinações costumam ser mais estáveis que qualquer modelo isolado.
- **Previsão probabilística avaliada por CRPS/pinball loss**, já que os intervalos são usados como estoque de segurança.
- **Previsão por produto** para a reposição de insumos específicos (leite, chocolate), com modelos para demanda intermitente nos produtos de baixo giro.
- **Experimento A/B de horário** (desenho na seção 20.5 do notebook principal) para medir a incrementalidade de forma causal.
- **Modelo global** (uma árvore ou rede para todas as máquinas) quando houver dezenas de séries.

## 5. Evoluções da arquitetura

1. **Curto prazo:** agendar `python -m src.pipeline` diariamente (cron/Prefect), gravar previsões versionadas e publicar um painel.
2. **Médio prazo:** registro de modelos (MLflow), monitoramento automático (seção 3 de `architecture.md`), *shadow evaluation* antes de promover modelos.
3. **Longo prazo:** ingestão direta dos logs das máquinas, *feature store* com eventos operacionais, modelo global para a frota.

## 6. Próximos passos para operacionalização

| # | Passo | Responsável sugerido | Pré-requisito |
|---|---|---|---|
| 1 | Validar com a operação o que aconteceu em mai/2024, jul/2024, nov/2024 e fev/2025 | negócio + dados | — |
| 2 | Criar o cadastro de eventos operacionais e de indisponibilidade | operação | 1 |
| 3 | Agendar o pipeline diário e publicar previsões e intervalos | engenharia de dados | — |
| 4 | Acompanhar por 4–8 semanas em *shadow mode* (previsão × real) antes de usar em decisões | dados | 3 |
| 5 | Integrar estoque e reposições para medir rupturas | operação + engenharia | 3 |
| 6 | Reavaliar a M2 com 3+ meses de histórico e incluí-la no backtest | dados | 4 |
