# Coffee Sales: previsão de demanda e faturamento por máquina

Solução de Ciência de Dados ponta a ponta para **prever as vendas diárias e a receita diária dos próximos 14 dias** de duas máquinas de café (M1 e M2) e da operação consolidada. Dos mesmos dados também saem conclusões sobre público, cardápio e operação.

**Conteúdo do repositório:**
- entendimento do negócio;
- EDA com tratamento justificado de ausentes, zeros, outliers e duplicados;
- testes formais de quebra estrutural;
- hipóteses de negócio (nova clientela e canibalização);
- atributos sem vazamento;
- 2 baselines e 4 modelos candidatos em 6 séries;
- validação temporal com backtest de 11 origens;
- previsões finais, arquitetura de produção e apresentação executiva.

> **Dados:** dataset Kaggle [`ihelon/coffee-sales`](https://www.kaggle.com/datasets/ihelon/coffee-sales), versão 21 (`index_1.csv` = Máquina 1, `index_2.csv` = Máquina 2). Os arquivos estão versionados em `data/raw/`, e o `MANIFEST.json` guarda o SHA-256 de cada um, conferido a cada execução.

## Sumário

1. [Resumo do projeto](#1-resumo-do-projeto)
2. [Estrutura do repositório](#2-estrutura-do-repositório)
3. [Dependências](#3-dependências)
4. [Instruções para execução](#4-instruções-para-execução)
5. [Reprodutibilidade](#5-reprodutibilidade)
6. [Boas práticas de engenharia de software](#6-boas-práticas-de-engenharia-de-software)
7. [Documentação](#7-documentação)

## 1. Resumo do projeto

### O que é previsto

| Item | Definição |
|---|---|
| Observação | 1 dia de 1 máquina, em calendário contínuo. Dia sem registro dentro do período ativo vale 0, sem imputação. |
| Alvo 1: vendas | número de registros de venda no dia; cada registro é 1 produto vendido, com seu valor. Gráficos e código chamam isso de "transação". |
| Alvo 2: receita | soma dos valores (`money`) dos registros do dia, em unidades monetárias (u.m.) |
| Registro ≠ pedido | não há ID de pedido, então um pedido com vários itens vira vários registros |
| Séries | M1 e M2 previstas separadamente. O Total recomendado é a soma das duas previsões. |
| Histórico | M1: 01/03/2024–23/03/2025 (388 dias, 3.636 registros). M2: 08/02/2025–23/03/2025 (44 dias, 262 registros). |
| Horizonte | 14 dias (24/03–06/04/2025). Teste nos últimos 14 dias observados (10–23/03/2025). |

### Principais resultados

| Série | Modelo escolhido | WAPE no teste (10–23/03/2025) | MASE médio no backtest (11 origens) | Leitura |
|---|---|---|---|---|
| Máquina 1 · vendas | Holt-Winters (8 semanas, Anscombe) | 22,4% | 0,77 (o melhor) | 24% menos erro que o naive; não significativo após Holm (p = 0,07) |
| Máquina 1 · receita | Holt-Winters (8 semanas) | 20,4% | 0,82 | o melhor no teste |
| Máquina 2 · vendas | Random Forest regularizada | 41,1% | — (só 30 dias de treino) | perto do piso de ruído (~32%); previsão quase plana, com faixa larga |
| Máquina 2 · receita | Random Forest regularizada | 37,3% | — | idem |
| Total · vendas | SARIMAX (modelo independente do Total, com `maquinas_ativas` e `dummy_novos_clientes`) | 20,6% | 0,89 | significativamente melhor que o naive (p Holm = 0,001) |
| Total · receita | SARIMAX (modelo independente do Total) | 19,5% | 0,92 | significativamente melhor que o naive (p Holm = 0,005) |

**Total: modelo independente × soma das previsões individuais (teste de 10–23/03/2025)**

| Total | Modelo independente (SARIMAX) | Soma das previsões M1 + M2 |
|---|---|---|
| Vendas: MAE / WAPE | 4,13 / 20,6% | **3,59 / 18,0%** (13% menos erro) |
| Receita: MAE / WAPE | 117 / 19,5% | **107 / 17,8%** (8% menos erro) |

O Total é modelado diretamente, como série própria, e a soma das previsões por máquina é calculada à parte para comparação. Nesta janela, a soma errou menos. Por isso ela é a **recomendação operacional** para o número consolidado: a operação decide por máquina, e o modelo único achata padrões semanais opostos (M1 em dias úteis, M2 no fim de semana).

Duas ressalvas:
- Nas vendas, a soma tem RMSE maior (4,9 × 4,6).
- A evidência vem de uma única janela de 14 dias, porque a M2 só existe desde 08/02/2025.

Por isso o modelo independente do Total segue em paralelo, como alarme de coerência.

As tabelas completas estão em [reports/model_comparison/](reports/model_comparison/). As previsões dos próximos 14 dias (24/03–06/04/2025), com intervalos de 80% e 95%, estão em [reports/forecasts/](reports/forecasts/).

**Previsão para 24/03–06/04/2025:**
- M1: 210 vendas e 7.198 u.m.
- M2: 88 vendas e 2.316 u.m.
- Total (soma): 298 vendas e 9.514 u.m., cerca de 21 vendas e 680 u.m. por dia.

**Achados de negócio:**
- **Duas máquinas, não uma.** 13 de 13 testes rejeitam que `index_2` repita `index_1`: 0 de 262 vendas casadas, e o controle positivo detecta uma cópia de 5%.
- **A demanda muda de patamar, não de tendência.** Há quatro quebras estruturais, todas com p < 0,001: horário ampliado (mai/2024), após a redução de preços (jul/2024), após a alta de preços de outubro (nov/2024) e nova clientela (jan/2025). As quebras de preço são associação no tempo, não prova de causa.
- **O salto de 2025 é uma troca de clientela da M1.** 177 dos 196 cartões ativos depois de 30/01/2025 são novos, e os clientes antigos quase desaparecem. A M2 soma volume próprio, sem perda líquida na M1.
- **A ampliação do horário de 2024 somou demanda.** O total subiu 45% e o núcleo 10–19h ficou estável. A incrementalidade de 0,84 (IC 0,34–1,23) não permite excluir canibalização parcial.
- **A receita depende de poucos produtos e de um público que gasta menos.**
  - 5 de 33 produtos fazem 80% da receita.
  - 17 dos 30 itens da M2 venderam até 5 vezes em 44 dias.
  - O público de 2025 da M1 gasta 3,6% menos por compra, com a mesma tabela de preços, porque troca Latte por Americano.
  - A M2 tem ticket 16% menor.
- **Padrão semanal só em 2025.** Antes de 30/01/2025, a M1 não tinha efeito de dia da semana. Depois, o fim de semana vende 55% menos que os dias úteis. O periodograma com teste de Fisher, aplicado só a 2025, encontra período 7 (p = 0,0003; notebook 05, seção 3.5).
- **Parte do erro é irredutível.** Mesmo conhecendo o nível exato, o ruído de contagem impõe um WAPE de ~17–32% (M1 21%, M2 32%, Total 17%; vendas, últimos 28 dias do treino). Os modelos escolhidos ficam perto desse piso.

Não há custo, margem nem estoque nos dados, então nenhuma recomendação tem ganho financeiro estimado. As ações propostas são pilotos medidos contra a previsão.

## 2. Estrutura do repositório

```text
case_coffee_sales/
├── config/config.yaml           # todos os parâmetros: datas, janelas, grades, pesos da seleção, sementes, paleta
├── data/
│   ├── raw/                     # CSVs originais (imutáveis) + MANIFEST.json com SHA-256
│   └── processed/               # transações padronizadas e séries diárias (artefatos/ é gerado e não versionado)
├── src/                         # código-fonte (pacote Python)
│   ├── data/                    # carga, contrato de dados, conferência de hash, séries diárias
│   ├── eda/                     # descritiva, ausentes, zeros, outliers, duplicados, produtos, tendência e sazonalidade
│   ├── structural_breaks/       # identificação das máquinas, testes de quebra, clientes, canibalização
│   ├── features/                # calendário, defasagens, dummies de quebra, pipeline sem vazamento
│   ├── models/                  # baselines, estatísticos (HW, SARIMAX, Poisson/NB2), Random Forest, seleção
│   ├── validation/              # métricas, backtesting, comparação e seleção final
│   ├── visualization/plots.py   # gráficos padronizados
│   ├── settings.py              # leitura da configuração e caminhos
│   ├── logging_config.py        # logs em arquivo e no console
│   └── pipeline.py              # execução ponta a ponta (9 etapas)
├── tests/                       # pytest: dados, vazamento, modelos, previsões, EDA de negócio
├── notebooks/                   # 01 entendimento · 02 EDA · 03 quebras · 04 atributos · 05 tendência/sazonalidade · 06 modelagem · 07 validação (*)
├── models/                      # modelos finais (joblib) + metadados (hash de dados e configuração) + registro
├── reports/                     # tabelas (CSV) e figuras (PNG 300 dpi + PDF) de cada etapa
│   └── apresentacao/            # apresentação executiva (PPTX) e o gerador em fonte/
├── docs/                        # negócio · metodologia · arquitetura · model card · limitações
├── .github/workflows/tests.yml  # CI: testes + pipeline em modo rápido a cada push
├── Makefile                     # atalhos: ambiente, pipeline, testes, notebooks, apresentação
├── requirements.txt             # dependências Python com versões exatas
└── .python-version              # 3.13
```

(*) O notebook 05, de tendência, sazonalidade e ciclo, foi acrescentado à estrutura pedida no enunciado. Por isso modelagem e validação ficaram como 06 e 07.

O pipeline tem 9 etapas, cada uma executável isoladamente:

`dados → identificacao → eda → quebras → features → modelagem → backtest → avaliacao → previsoes`

Os notebooks não reimplementam lógica: eles chamam `src/` e leem os artefatos do pipeline, então análise e produção usam o mesmo código.

## 3. Dependências

| Componente | Versão | Para quê |
|---|---|---|
| **Python** | 3.13 (resultados gerados com 3.13.7) | todo o projeto |
| numpy / pandas / scipy | 2.5.4 / 2.3.3 / 1.18.1 | base numérica e tabular |
| statsmodels | 0.15.0 | Holt-Winters, SARIMAX, Poisson/NB2, testes estatísticos |
| pmdarima | 2.1.1 | busca de ordens ARIMA |
| scikit-learn | 1.9.1 | Random Forest |
| ruptures | 1.1.10 | detecção de quebras (Binseg) |
| pymannkendall | 1.4.3 | teste de tendência de Mann-Kendall |
| matplotlib / seaborn / pillow | 3.11.2 / 0.13.2 / 13.0.0 | gráficos e leitura das figuras |
| joblib | 1.6.0 | paralelismo do backtest e serialização dos modelos |
| PyYAML | 6.0.3 | configuração |
| kagglehub | 1.0.3 | só se `data/raw/` estiver vazia (baixa a versão 21 do Kaggle) |
| jupyter / nbconvert / ipykernel | 1.1.1 / 7.17.2 / 7.4.0 | notebooks |
| pytest | 9.2.0 | testes |
| pyarrow, python-pptx | 26.0.1, 1.0.2 | opcionais: backend do pandas e inspeção do PPTX |
| **Node.js** (opcional) | ≥ 18; pptxgenjs 3.12.0 e jszip 3.10.2 (`package-lock.json`) | só para regenerar a apresentação |

Todas as versões Python estão fixadas em [requirements.txt](requirements.txt), e as do Node em [reports/apresentacao/fonte/package-lock.json](reports/apresentacao/fonte/package-lock.json). Não é preciso GPU nem serviço de nuvem.

## 4. Instruções para execução

Os comandos abaixo valem para macOS/Linux e partem da raiz do repositório. Entre parênteses está o alvo equivalente do `Makefile`.

```bash
# 1. Ambiente Python 3.13 com as versões fixadas                      (make ambiente)
python3.13 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Pipeline completo, ~10 min                                       (make pipeline)
#    dados → identificação → EDA → quebras → atributos → modelos → backtest → seleção → previsões
python -m src.pipeline

#    variações
python -m src.pipeline --rapido                  # sem o backtest de 11 origens, ~5 min  (make rapido)
python -m src.pipeline --etapas dados eda        # só as etapas escolhidas
python -m src.validation.meta_decisao             # opcional: refaz a meta-decisão da regra da RF, ~5 min (make meta-decisao)

# 3. Testes                                                           (make testes / make testes-rapidos)
pytest -m "not slow"                 # segundos; não treina a Random Forest
pytest                               # todos

# 4. Notebooks: rodar depois do pipeline, que gera data/processed/artefatos/   (make notebooks)
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# 5. Apresentação: opcional, requer Node.js                           (make apresentacao)
cd reports/apresentacao/fonte
python dados_deck.py                 # extrai de reports/ todos os números e figuras do deck
npm ci                               # instala as versões do package-lock.json
node gerar_deck.js                   # grava ../coffee_sales_executiva.pptx
```

**Saídas:**

| Pasta | Conteúdo |
|---|---|
| `reports/` | tabelas CSV e figuras PNG/PDF de cada etapa |
| `models/` | modelos finais e seus metadados |
| `data/processed/` | séries diárias |
| `logs/pipeline.log` | log da execução |

Os dados já estão em `data/raw/`, então não é preciso acesso à internet. O `kagglehub` só entra se os CSVs forem apagados, e o download pode exigir login no Kaggle.

**Apresentação:** [coffee_sales_executiva.pptx](reports/apresentacao/coffee_sales_executiva.pptx) tem 16 slides principais (~30 min) e apêndice técnico navegável (A1–A12), gerado por `reports/apresentacao/fonte/gerar_deck.js`.
- Nenhum número do deck é digitado à mão quando há uma tabela em `reports/` que o contenha.

## 5. Reprodutibilidade

**O que garante os mesmos resultados:**

| Fonte de variação | Como foi controlada |
|---|---|
| Dados | CSVs versionados. O SHA-256 é **conferido** contra o `MANIFEST.json` a cada execução; arquivo divergente interrompe o pipeline com erro (`src/data/load_data.py`). |
| Bibliotecas | versões exatas em `requirements.txt` e `package-lock.json`; Python 3.13 em `.python-version` |
| Aleatoriedade | sementes fixas em `config/config.yaml` (`projeto.semente` = 2024, `quebras.semente_bootstrap` = 42) e nos geradores (`np.random.default_rng`, `random_state` da Random Forest) |
| Paralelismo | o backtest roda em processos separados (`joblib`), que devolvem os resultados na ordem das origens; como as sementes são fixas, o resultado não depende de qual processo termina primeiro |
| Decisões antes do teste | a regra de seleção da Random Forest (1 erro-padrão × mínimo) foi decidida no backtest das 10 origens anteriores ao teste; `python -m src.validation.meta_decisao` refaz essa comparação |
| Parâmetros | nenhum número mágico no código: datas, janelas, grades e pesos ficam no `config.yaml`, cujo hash entra nos metadados de cada modelo |
| Vazamento | atributos calculados só com dados até a véspera; dois testes automáticos no CI trocam o futuro por ruído e exigem resultado idêntico |

**Verificação feita (10/11/2026).** O repositório foi testado como se outra pessoa o clonasse:
1. Só os arquivos versionados foram copiados para uma pasta nova.
2. Todas as saídas (`reports/`, `models/`, `data/processed/`) foram apagadas.
3. Um ambiente Python 3.13 novo foi criado com `pip install -r requirements.txt`.

Nessa cópia:
- o pipeline completo rodou em ~10–13 min;
- os 82 testes passaram;
- os 7 notebooks executaram sem erro;
- a apresentação foi regenerada com `npm ci`.

**Resultado da comparação com o repositório:**
- 98 das 101 tabelas de resultado (CSV/JSON) saíram **idênticas byte a byte**.
- Nas outras 3, só diferem colunas de relógio: `tempo_s` (tempo de execução de cada modelo) e `gerado_em`.
- O `dados_deck.json` saiu idêntico, e os slides, gráficos, tema e imagens do PPTX também.
- A meta-decisão da regra da Random Forest (`python -m src.validation.meta_decisao`) reproduziu byte a byte o CSV versionado.

O teste encontrou três problemas, todos corrigidos:
- `pymannkendall` faltava no `requirements.txt`.
- Os intervalos do Holt-Winters com sazonalidade multiplicativa eram simulados sem semente. Agora o gerador é fixo, com teste de regressão (`test_holt_winters_reprodutivel`).
- O gerador da apresentação dependia de caminhos da máquina do autor.

**Limites honestos:**
- Em outro sistema operacional ou processador (por exemplo, Linux x86 em vez de macOS arm64), a biblioteca de álgebra linear pode mudar as últimas casas decimais dos ajustes numéricos (SARIMAX, Holt-Winters). As métricas e as escolhas de modelo devem ser as mesmas, mas não há garantia de igualdade bit a bit entre plataformas.
- Colunas de relógio mudam a cada execução sem afetar resultados: `gerado_em` nos metadados dos modelos e `tempo_s` nas tabelas de backtest. Os PDFs das figuras e o PPTX também guardam data de criação, e o PPTX tem IDs de seção aleatórios.

## 6. Boas práticas de engenharia de software

| Prática | Evidência no repositório |
|---|---|
| Código modular, separado por domínio | `src/` com 7 subpacotes (dados, EDA, quebras, atributos, modelos, validação, visualização); notebooks só orquestram |
| Documentação no código | 220 de 251 funções públicas com docstring (Google style) e 240 com anotação de tipo de retorno; docstring de módulo explicando as decisões |
| Configuração centralizada | `config/config.yaml`; o código não contém números mágicos |
| Pipeline reprodutível e idempotente | `python -m src.pipeline`, 9 etapas encadeadas, cada uma executável isoladamente (`--etapas`) |
| Testes automatizados | 82 testes em `tests/` (pytest): contrato e hash dos dados, reprodutibilidade dos modelos, invariantes de negócio (Total = M1 + M2; zero de vendas ⇔ zero de receita), ausência de vazamento, modelos, previsões e análises de negócio. Testes lentos marcados com `@pytest.mark.slow` |
| Integração contínua | GitHub Actions ([.github/workflows/tests.yml](.github/workflows/tests.yml)): a cada push, instala as dependências, roda os testes e executa o pipeline em modo rápido |
| Contrato de dados | `validar_bruto`: colunas, tipos, domínio de `cash_type`, `money > 0`, coerência `date` × `datetime`; violação gera erro explícito, sem descarte silencioso |
| Rastreabilidade | dados com SHA-256; modelos em `joblib` com metadados (série, alvo, especificação, janela de treino, hash dos dados e da configuração) e `models/registro_modelos.csv` |
| Logging | `src/logging_config.py`; log de cada etapa, com duração, em `logs/pipeline.log` |
| Tratamento de dados explícito | nada é imputado ou removido em silêncio; cada decisão sobre ausentes, zeros, outliers e duplicados está justificada (notebook 02 e `docs/methodology.md`) |
| Avaliação honesta | baselines, validação interna dentro do treino, backtest de 11 origens, teste nunca usado para ajustar, Diebold-Mariano com correção de Holm, piso de ruído |
| Versionamento | Git; `.gitignore` exclui artefatos gerados, logs, caches e ambientes |
| Arquitetura de produção | batch diário, monitoramento, drift, retreino e shadow mode em [docs/architecture.md](docs/architecture.md), separando o que está implementado do que é proposta |

## 7. Documentação

| Documento | Conteúdo |
|---|---|
| [docs/business_understanding.md](docs/business_understanding.md) | problema, oportunidade, objetivos, KPIs, hipóteses, limitações dos dados |
| [docs/methodology.md](docs/methodology.md) | decisões e justificativas: dados, quebras, atributos, modelos, alternativas descartadas, validação, seleção |
| [docs/model_card.md](docs/model_card.md) | modelos escolhidos, desempenho, uso pretendido, limitações e impacto de negócio |
| [docs/architecture.md](docs/architecture.md) | arquitetura de produção, monitoramento, drift, retreino, versionamento, escalabilidade |
| [docs/limitations.md](docs/limitations.md) | limitações, riscos, dados adicionais, melhorias e próximos passos |
| [reports/apresentacao/](reports/apresentacao/) | apresentação executiva (PPTX, ~30 min) e seu gerador |
| [docs/rastreabilidade_requisitos.md](docs/rastreabilidade_requisitos.md) | cada item do enunciado do desafio e onde ele está atendido no repositório |

### Origem do trabalho

Este projeto reorganiza e reaproveita análises exploratórias anteriores, que **não fazem parte deste repositório**:
- `coffee_sales_analise_estatistica_series_temporais.ipynb`: EDA, quebras, horário e canibalização;
- `modelagens_testes.ipynb`: protocolo de modelagem, backtest de 11 origens e descarte de modelos;
- `teste_index2_nova_maquina.ipynb`: identificação das máquinas.

As metodologias desses notebooks foram portadas para `src/` sem alteração (as sementes também foram mantidas) e são testadas aqui.
