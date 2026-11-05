# -*- coding: utf-8 -*-
"""
logging.py — Centralised logging setup.
Correções aplicadas:
  - get_logger() idempotente: se o logger já tem
    handlers, apenas o nível é atualizado.
  - propagate=False para evitar duplicação com
    root logger.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Retorna um logger configurado.
    Escreve no stdout e opcionalmente em log_file.
    Parâmetros
    ----------
    name     : nome do logger
               (tipicamente __name__).
    level    : nível de log
               ("DEBUG", "INFO", "WARNING",
               "ERROR").
               Usa INFO se nível não reconhecido.
    log_file : caminho opcional para arquivo de
               log. Diretórios pai são criados
               automaticamente.
    Notas
    -----
    Idempotente: se o logger já tem handlers,
    apenas o nível é atualizado e o logger é
    retornado. Isso previne entradas duplicadas
    quando get_logger é chamado múltiplas vezes
    com o mesmo name.
    propagate=False previne duplicação quando o
    root logger tem handlers configurados
    (ex: basicConfig).
    Limitação: se a primeira chamada não incluir
    log_file e uma segunda chamada incluir, o
    arquivo não será adicionado (logger já tem
    handlers). Para garantir log em arquivo,
    passar log_file na primeira chamada ou usar
    logging.basicConfig.
    """
    logger = logging.getLogger(name)
    resolved_level = getattr(
        logging,
        level.upper(),
        logging.INFO,
    )
    if logger.handlers:
        logger.setLevel(resolved_level)
        return logger
    logger.setLevel(resolved_level)
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | "
        "%(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(resolved_level)
    logger.addHandler(ch)
    # File handler (opcional)
    if log_file:
        try:
            Path(log_file).parent.mkdir(
                parents=True, exist_ok=True
            )
            fh = logging.FileHandler(
                log_file, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            fh.setLevel(resolved_level)
            logger.addHandler(fh)
        except OSError as e:
            logger.warning(
                "Could not create log file "
                "'%s': %s — "
                "logging to stdout only.",
                log_file, e,
            )
    return logger