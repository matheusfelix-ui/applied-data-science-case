# Atalhos de execução (GNU make). Cada alvo equivale a um comando do README.
PY ?= .venv/bin/python
PYABS := $(abspath $(PY))

.PHONY: ambiente pipeline rapido meta-decisao testes testes-rapidos notebooks apresentacao tudo

ambiente:        ## cria o .venv (Python 3.13) e instala as versões fixadas
	python3.13 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

pipeline:        ## pipeline completo: dados → EDA → quebras → atributos → modelos → backtest → seleção → previsões (~10 min)
	$(PY) -m src.pipeline

rapido:          ## pipeline sem o backtest de 11 origens (~5 min)
	$(PY) -m src.pipeline --rapido

meta-decisao:    ## refaz a comparação das regras de seleção da RF nas origens anteriores ao teste (~5 min)
	$(PY) -m src.validation.meta_decisao

testes-rapidos:  ## testes sem treinar a Random Forest (segundos)
	$(PY) -m pytest -m "not slow" -q

testes:          ## todos os testes
	$(PY) -m pytest -q

notebooks:       ## reexecuta os notebooks (depois do pipeline)
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

apresentacao:    ## regenera o PPTX a partir de reports/ (requer Node.js >= 18)
	cd reports/apresentacao/fonte && "$(PYABS)" dados_deck.py && npm ci && node gerar_deck.js

tudo: pipeline testes notebooks apresentacao
