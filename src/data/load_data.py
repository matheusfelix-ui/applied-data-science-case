"""Carga e validação dos dados brutos (``data/raw/index_1.csv`` e ``index_2.csv``).

Os arquivos brutos são a versão 21 do dataset Kaggle ``ihelon/coffee-sales``. Eles são copiados
uma vez para ``data/raw/`` (com SHA-256 em ``MANIFEST.json``) para que o projeto seja
reprodutível sem acesso à internet; se a pasta estiver vazia, o kagglehub é usado como fonte.

Cada arquivo recebe o identificador da máquina definido em ``config.yaml`` (``index_1`` → Máquina 1,
``index_2`` → Máquina 2). A atribuição é sustentada pelos testes de
``src/structural_breaks/machine_identification.py``.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from src.logging_config import obter_logger
from src.settings import carregar_config, caminho

log = obter_logger(__name__)


class ErroValidacaoDados(ValueError):
    """Dados de entrada fora do contrato esperado."""


def sha256(arquivo: Path) -> str:
    """Hash SHA-256 de um arquivo (rastreabilidade dos dados de entrada)."""
    h = hashlib.sha256()
    with open(arquivo, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 16), b""):
            h.update(bloco)
    return h.hexdigest()


def garantir_dados_brutos(config: dict | None = None) -> Path:
    """Garante que os CSVs brutos estejam em ``data/raw`` e grava o manifesto de hashes.

    Se algum arquivo faltar, baixa a versão fixada do Kaggle via kagglehub e copia para
    ``data/raw``. Os arquivos existentes nunca são sobrescritos.

    Quando ``MANIFEST.json`` já existe (ele é versionado no Git), os hashes são **conferidos**,
    e não regravados: se um arquivo diferir do manifesto, a carga é interrompida. Assim, quem
    clonar o repositório roda o pipeline exatamente sobre os mesmos bytes.

    Raises:
        ErroValidacaoDados: se o SHA-256 de algum arquivo divergir do manifesto versionado.
    """
    config = config or carregar_config()
    pasta = caminho("dados_brutos", config)
    arquivos = list(config["dados"]["arquivos"])
    faltantes = [a for a in arquivos if not (pasta / a).exists()]
    if faltantes:
        import kagglehub  # import tardio: só necessário sem os dados locais

        log.info("Arquivos ausentes em data/raw (%s); baixando %s", faltantes, config["dados"]["kaggle_dataset"])
        origem = Path(kagglehub.dataset_download(config["dados"]["kaggle_dataset"]))
        for a in faltantes:
            encontrados = list(origem.rglob(a))
            if not encontrados:
                raise FileNotFoundError(f"{a} não encontrado no download do Kaggle ({origem})")
            shutil.copy2(encontrados[0], pasta / a)
    manifesto = {a: {"sha256": sha256(pasta / a), "bytes": (pasta / a).stat().st_size} for a in arquivos}
    arquivo_manifesto = pasta / "MANIFEST.json"
    if arquivo_manifesto.exists():
        esperado = json.loads(arquivo_manifesto.read_text(encoding="utf-8"))["arquivos"]
        divergentes = [a for a in arquivos if a in esperado and esperado[a]["sha256"] != manifesto[a]["sha256"]]
        if divergentes:
            raise ErroValidacaoDados(
                f"data/raw difere do MANIFEST.json (SHA-256) em {divergentes}: os resultados não seriam reprodutíveis. "
                "Restaure os arquivos originais (versão 21 do Kaggle) ou apague o MANIFEST.json se a troca for intencional.")
        if all(a in esperado for a in arquivos):
            log.info("Dados brutos conferidos com o MANIFEST.json (SHA-256)")
            return pasta
    arquivo_manifesto.write_text(json.dumps({"fonte": config["dados"]["kaggle_dataset"],
                                             "arquivos": manifesto}, indent=2), encoding="utf-8")
    return pasta


def validar_bruto(d: pd.DataFrame, nome: str, config: dict) -> None:
    """Valida o contrato de um arquivo bruto antes de qualquer transformação.

    Regras: colunas obrigatórias presentes; ``datetime`` interpretável; ``money`` numérico e
    positivo; ``cash_type`` dentro do domínio. ``card`` é opcional (``index_2`` não tem a coluna).

    Raises:
        ErroValidacaoDados: com a lista de violações encontradas.
    """
    erros = []
    faltam = [c for c in config["dados"]["colunas_obrigatorias"] if c not in d.columns]
    if faltam:
        erros.append(f"colunas ausentes: {faltam}")
    else:
        dt = pd.to_datetime(d["datetime"], errors="coerce", format="mixed")
        if dt.isna().any():
            erros.append(f"{int(dt.isna().sum())} valores de datetime inválidos")
        dinheiro = pd.to_numeric(d["money"], errors="coerce")
        if dinheiro.isna().any():
            erros.append(f"{int(dinheiro.isna().sum())} valores de money não numéricos")
        if (dinheiro <= 0).any():
            erros.append(f"{int((dinheiro <= 0).sum())} valores de money não positivos")
        invalidos = set(d["cash_type"].astype(str).str.strip().unique()) - set(config["dados"]["meios_pagamento_validos"])
        if invalidos:
            erros.append(f"cash_type fora do domínio: {sorted(invalidos)}")
        data_texto = pd.to_datetime(d["date"], errors="coerce")
        if (data_texto.dt.normalize() != dt.dt.normalize()).any():
            erros.append("coluna date diverge da data do timestamp")
    if len(d) == 0:
        erros.append("arquivo vazio")
    if erros:
        for e in erros:
            log.error("%s: %s", nome, e)
        raise ErroValidacaoDados(f"{nome}: " + "; ".join(erros))


def carregar_brutos(config: dict | None = None) -> pd.DataFrame:
    """Lê, valida e empilha os arquivos brutos, marcando a origem e a máquina de cada registro.

    Returns:
        DataFrame com as colunas originais (``card`` ausente vira NaN na Máquina 2) mais
        ``arquivo_origem`` e ``maquina``.
    """
    config = config or carregar_config()
    pasta = garantir_dados_brutos(config)
    partes = []
    for nome, maquina in config["dados"]["arquivos"].items():
        d = pd.read_csv(pasta / nome)
        d.columns = d.columns.str.strip().str.lower()
        validar_bruto(d, nome, config)
        d["arquivo_origem"] = nome
        d["maquina"] = int(maquina)
        partes.append(d)
        log.info("%s: %d registros válidos (máquina %d)", nome, len(d), maquina)
    return pd.concat(partes, ignore_index=True)
