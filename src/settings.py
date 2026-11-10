"""Carregamento da configuração do projeto (``config/config.yaml``).

A configuração é lida uma única vez e exposta como dicionário. Os caminhos relativos
são resolvidos a partir da raiz do projeto, de modo que o código funcione a partir de
qualquer diretório de trabalho (pipeline, notebooks ou testes).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
CONFIG_PADRAO = RAIZ_PROJETO / "config" / "config.yaml"


@lru_cache(maxsize=4)
def carregar_config(caminho: str | Path | None = None) -> dict[str, Any]:
    """Lê o YAML de configuração e valida a presença das seções obrigatórias.

    Args:
        caminho: arquivo YAML alternativo; por padrão, ``config/config.yaml``.

    Returns:
        Dicionário com a configuração.

    Raises:
        FileNotFoundError: se o arquivo não existir.
        KeyError: se faltar alguma seção obrigatória.
    """
    caminho = Path(caminho) if caminho else CONFIG_PADRAO
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {caminho}")
    with open(caminho, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    obrigatorias = ["projeto", "caminhos", "dados", "series", "divisao", "quebras", "features", "modelos",
                    "selecao", "visualizacao", "logging"]
    faltantes = [s for s in obrigatorias if s not in config]
    if faltantes:
        raise KeyError(f"Seções ausentes na configuração: {faltantes}")
    return config


def caminho(chave: str, config: dict | None = None) -> Path:
    """Caminho absoluto de uma entrada de ``caminhos`` (cria o diretório se necessário)."""
    config = config or carregar_config()
    p = RAIZ_PROJETO / config["caminhos"][chave]
    p.mkdir(parents=True, exist_ok=True)
    return p


def pasta_relatorio(*partes: str) -> Path:
    """Subpasta de ``reports/`` (criada se não existir)."""
    p = caminho("relatorios").joinpath(*partes)
    p.mkdir(parents=True, exist_ok=True)
    return p
