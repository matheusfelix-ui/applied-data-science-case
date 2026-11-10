"""Configuração de logging: console + arquivo (``logs/pipeline.log``).

Uso:
    from src.logging_config import obter_logger
    log = obter_logger(__name__)
"""
from __future__ import annotations

import logging
import warnings

from src.settings import RAIZ_PROJETO, carregar_config

_CONFIGURADO = False
FORMATO = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configurar_logging(nivel: str | None = None, arquivo: str | None = None) -> None:
    """Configura o logger raiz do projeto (idempotente).

    Os avisos de convergência repetitivos de statsmodels/pmdarima são silenciados para manter
    o log legível; a convergência dos modelos é registrada explicitamente pelo código.
    """
    global _CONFIGURADO
    if _CONFIGURADO:
        return
    config = carregar_config()
    nivel = nivel or config["logging"]["nivel"]
    arquivo = RAIZ_PROJETO / (arquivo or config["logging"]["arquivo"])
    arquivo.parent.mkdir(parents=True, exist_ok=True)

    raiz = logging.getLogger("src")
    raiz.setLevel(nivel)
    raiz.propagate = False
    formatador = logging.Formatter(FORMATO, datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatador)
    em_arquivo = logging.FileHandler(arquivo, encoding="utf-8")
    em_arquivo.setFormatter(formatador)
    raiz.handlers = [console, em_arquivo]

    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
    warnings.filterwarnings("ignore", module="pmdarima")
    _CONFIGURADO = True


def obter_logger(nome: str) -> logging.Logger:
    """Logger filho de ``src`` já configurado."""
    configurar_logging()
    if not nome.startswith("src"):
        nome = f"src.{nome}"
    return logging.getLogger(nome)
