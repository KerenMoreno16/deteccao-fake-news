# -*- coding: utf-8 -*-
"""
reproducibility.py — Set and register all random
seeds; log environment info.
Correções aplicadas:
  - "sklearn" removido de get_library_versions()
    (duplicata de "scikit-learn").
  - Normalização de "pyyaml" simplificada.
  - datetime.now(timezone.utc) substitui
    datetime.utcnow() (deprecated Python 3.12+).
  - Chave "pyyaml" usada em vez de "yaml" para
    consistência com o nome do pacote pip.
"""
import hashlib
import importlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
def set_seed(seed: int = 42) -> None:
    """
    Define seeds para Python, NumPy, PyTorch e
    TensorFlow (se instalados).
    Parâmetros
    ----------
    seed : valor inteiro da seed (padrão 42).
    Notas
    -----
    - PYTHONHASHSEED é definido para
      reprodutibilidade de estruturas hash.
    - PyTorch e TensorFlow são configurados apenas
      se instalados.
    - torch.backends.cudnn.deterministic=True pode
      reduzir performance GPU mas garante
      reprodutibilidade.
    - np.random.seed() afeta o gerador legado —
      necessário para sklearn e scipy que usam
      RandomState internamente. Para novo código
      NumPy, preferir np.random.default_rng(seed).
    """
    random.seed(seed)
    # Gerador legado — necessário para sklearn/scipy
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
def get_library_versions() -> Dict[str, str]:
    """
    Retorna versões das principais bibliotecas.
    Bibliotecas não instaladas são registradas
    como "not installed" em vez de levantar
    ImportError.
    Notas
    -----
    - "sklearn" removido (duplicata de
      "scikit-learn").
    - Chave "pyyaml" usada para o módulo yaml,
      consistente com o nome do pacote pip.
    """
    # Mapeamento: nome_exibição -> nome_import
    libs = {
        "numpy":        "numpy",
        "pandas":       "pandas",
        "scikit-learn": "sklearn",
        "torch":        "torch",
        "transformers": "transformers",
        "datasets":     "datasets",
        "shap":         "shap",
        "lime":         "lime",
        "scipy":        "scipy",
        "matplotlib":   "matplotlib",
        "seaborn":      "seaborn",
        "pyyaml":       "yaml",
        "tqdm":         "tqdm",
        "joblib":       "joblib",
    }
    versions: Dict[str, str] = {}
    for display_name, import_name in libs.items():
        try:
            mod = importlib.import_module(
                import_name
            )
            versions[display_name] = getattr(
                mod, "__version__", "unknown"
            )
        except ImportError:
            versions[display_name] = (
                "not installed"
            )
    return versions
def get_gpu_info() -> Dict[str, Any]:
    """
    Retorna informações de disponibilidade de GPU.
    Retorna
    -------
    dict com:
        available : bool
        count     : int
        devices   : list of str
    """
    info: Dict[str, Any] = {
        "available": False,
        "count":     0,
        "devices":   [],
    }
    try:
        import torch
        info["available"] = (
            torch.cuda.is_available()
        )
        info["count"]     = (
            torch.cuda.device_count()
        )
        info["devices"]   = [
            torch.cuda.get_device_name(i)
            for i in range(info["count"])
        ]
    except ImportError:
        pass
    return info
def register_environment(
    seed: int,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Coleta e opcionalmente salva snapshot completo
    do ambiente de execução.
    Chama set_seed() para inicializar todos os
    geradores de números aleatórios.
    Parâmetros
    ----------
    seed        : seed a registrar e aplicar.
    output_path : se fornecido, salva snapshot
                  como JSON.
    Retorna
    -------
    dict com: timestamp, seed, python_version,
    platform, library_versions, gpu.
    """
    set_seed(seed)
    env: Dict[str, Any] = {
        # datetime.now(timezone.utc) substitui
        # datetime.utcnow() deprecated Python 3.12+
        "timestamp": (
            datetime.now(timezone.utc).isoformat()
        ),
        "seed":             seed,
        "python_version":   (
            platform.python_version()
        ),
        "platform":         platform.platform(),
        "library_versions": (
            get_library_versions()
        ),
        "gpu":              get_gpu_info(),
    }
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(
            parents=True, exist_ok=True
        )
        with open(
            out, "w", encoding="utf-8"
        ) as f:
            json.dump(
                env, f,
                indent=2,
                default=str,
            )
    return env
def file_hash(
    path: str,
    algorithm: str = "sha256",
) -> str:
    """
    Retorna o digest hexadecimal de um arquivo
    para rastreamento de proveniência de datasets.
    Parâmetros
    ----------
    path      : caminho para o arquivo.
    algorithm : algoritmo de hash
                (padrão "sha256").
    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    ValueError
        Se o algoritmo não for suportado.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"File not found for hashing: {path}"
        )
    try:
        h = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported hash algorithm "
            f"'{algorithm}'. "
            f"Use one of: "
            f"{sorted(hashlib.algorithms_available)}"
        ) from exc
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(65536), b""
        ):
            h.update(chunk)
    return h.hexdigest()