# -*- coding: utf-8 -*-
"""
config.py — Load and validate the YAML experiment
configuration.
"""
from pathlib import Path
from typing import Any, Dict
import yaml
from src.utils.logging import get_logger
logger = get_logger(__name__)
# Sentinel para distinguir "chave ausente" de None real
_MISSING = object()
def load_config(
    path: str = "configs/experiment.yaml",
) -> Dict[str, Any]:
    """
    Carrega configuração YAML e retorna como dict.
    Parâmetros
    ----------
    path : caminho para o arquivo YAML.
    Retorna
    -------
    dict com a configuração completa.
    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    ValueError
        Se o arquivo estiver vazio, contiver YAML
        inválido ou a estrutura não for um dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )
    with open(p, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"YAML inválido em '{path}': {exc}"
            ) from exc
    if cfg is None:
        raise ValueError(
            f"Config file is empty or invalid: "
            f"{path}"
        )
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config must be a YAML mapping (dict), "
            f"got {type(cfg).__name__}: {path}"
        )
    logger.info(
        "Config loaded: %s (%d top-level keys)",
        path, len(cfg),
    )
    return cfg
def get_nested(
    cfg: Dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """
    Navega chaves aninhadas de forma segura.
    Retorna default se qualquer chave estiver
    ausente ou se um valor intermediário não
    for um dict.
    Parâmetros
    ----------
    cfg     : dict de configuração.
    *keys   : sequência de chaves para navegar.
    default : valor retornado quando chave ausente.
    Notas
    -----
    Se o valor real for None, o retorno será None —
    indistinguível de "chave ausente" com
    default=None. Para distinguir os dois casos,
    use default=_MISSING e verifique
    ``result is _MISSING``.
    Exemplos
    --------
    >>> get_nested({"a": {"b": 1}}, "a", "b")
    1
    >>> get_nested({"a": {"b": 1}}, "a", "c",
    ...            default=0)
    0
    >>> get_nested({"a": 1}, "x",
    ...            default=_MISSING) is _MISSING
    True
    """
    cur = cfg
    for k in keys:
        if (
            not isinstance(cur, dict)
            or k not in cur
        ):
            return default
        cur = cur[k]
    return cur
def flatten_config(
    cfg: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Aplana dict de configuração aninhado em um
    único nível com chaves separadas por ponto.
    Listas são tratadas como valores escalares e
    não são recursivamente aplanadas. Apenas dicts
    são recursivamente aplanados.
    Exemplos
    --------
    >>> flatten_config(
    ...     {"a": {"b": 1, "c": 2}, "d": 3})
    {'a.b': 1, 'a.c': 2, 'd': 3}
    >>> flatten_config(
    ...     {"tfidf": {"ngram_range": [1, 2]}})
    {'tfidf.ngram_range': [1, 2]}
    """
    items: Dict[str, Any] = {}
    for k, v in cfg.items():
        full_key = (
            f"{prefix}.{k}" if prefix else k
        )
        if isinstance(v, dict):
            items.update(
                flatten_config(v, prefix=full_key)
            )
        else:
            items[full_key] = v
    return items