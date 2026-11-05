"""
run_all_experiments.py - Orquestrador de todo o pipeline experimental.
Versão 5.2 - Multi-seed (10 execuções independentes):
Última atualização: 20-10-2026
  SEEDS = [42, 43, ..., 51]
  Para cada seed:
    - O dataset é re-dividido com random_split(seed=seed).
    - Todos os modelos são (re-)inicializados com random_state=seed.
    - Os resultados são salvos individualmente com sufixo _seed{N}.
  Após todas as seeds:
    - Os resultados individuais são agregados (média ± desvio padrão amostral).

    6 datasets:
  PT-BR: FakeTrue.Br | Fake.br-Corpus | FakeRecogna
  EN:    FakeNewsNet-PolitiFact | PHEME | CONSTRAINT-COVID19
Modelos:
  - Logistic Regression + TF-IDF
  - MLP + TF-IDF
  - BiLSTM + FastText embeddings
  - BERTimbau (pt) / bert-base-uncased (en)
Experimentos:
  A - Intra-dataset
  B - Cross-dataset
  C - Combined PT-BR / EN
  D - Temporal
Uso:
    python run_all_experiments.py
    python run_all_experiments.py --skip-bert
    python run_all_experiments.py --dataset fakebr
    python run_all_experiments.py --seeds 42 43
"""

import json
import os
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from src.utils.config import load_config, get_nested
from src.utils.logging import get_logger
from src.utils.reproducibility import (
    register_environment,
    set_seed,
    file_hash,
)
from src.utils.constants import (
    PT_DATASETS,
    EN_DATASETS,
    TEMPORAL_DATASETS,
)
from src.data.loaders import load_dataset
from src.data.cleaning import clean_text
from src.data.validation import (
    assert_valid,
    print_dataset_report,
    validate_metrics_availability,
)
from src.data.deduplication import (
    text_hash,
    remove_exact_duplicates,
    detect_cross_dataset_leakage,
    near_duplicate_groups,
    remove_near_duplicates,
)
from src.data.splitting import random_split
from src.experiments.cross_dataset import (
    run_all_cross_dataset,
)
from src.experiments.temporal import (
    run_temporal_experiment,
)
from src.experiments.train import run_experiment
from src.experiments.statistical_tests import (
    run_statistical_tests_from_aggregation,
)

from src.utils.aggregation import (
    aggregate_all_experiments,
)
import re as _re
from sklearn.metrics import auc, roc_curve
import matplotlib
matplotlib.use("Agg")  # backend sem display
import matplotlib.pyplot as plt
from collections import defaultdict

# Logging
Path("results/logs").mkdir(
    parents=True, exist_ok=True
)
_RUN_TIMESTAMP = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)
logger = get_logger(
    __name__,
    log_file=(
        f"results/logs/run_{_RUN_TIMESTAMP}.log"
    ),
)

# Constants
SEEDS: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
MODELS: List[str] = [
    "logistic_regression",
    "mlp",
    "bilstm",
    "bertimbau",
]

DATASET_MAP: Dict[str, str] = {
    "faketruebr":         "data/raw/faketruebr",
    "fakebr":             "data/raw/fakebr",
    "fakerecogna":        "data/raw/fakerecogna",
    "fakenewsnet":        "data/raw/fakenewsnet",
    "constraint_covid19": (
        "data/raw/constraint_covid19"
    ),
    "pheme":              "data/raw/pheme",
}

_MODEL_ABBREV: Dict[str, str] = {
    "BERTimbau / BERT-base": "BERT",
    "LR + TF-IDF":           "LR",
    "MLP + TF-IDF":          "MLP",
    "BiLSTM":                "BiLSTM",
}

# Limite de arquivos para file_hash
_MAX_FILES_TO_HASH: int = 100
# Extensões de dados tabulares
_TABULAR_EXTENSIONS: Set[str] = {
    ".csv", ".parquet", ".xlsx",
}
# Chaves do YAML exclusivas do BiLSTM
_BILSTM_ONLY_KEYS: Set[str] = {
    "fasttext",
    "fasttext_paths",
    "fasttext_path",
}
# Chaves a remover sempre de model_cfg
_ALWAYS_REMOVE_KEYS: Set[str] = {
    "param_grid",
}
# Timeout para scripts de pós-processamento
_POSTPROCESS_TIMEOUT: int = 600

# Constantes para STEP 7 (Text length stats + ROC curves)
# Mapeamento nome interno -> nome de exibição
# para curvas ROC e tabela de comprimento
_MODEL_DISPLAY: Dict[str, str] = {
    "bertimbau":           "BERTimbau / BERT-base",
    "bert_base_uncased":   "BERTimbau / BERT-base",
    "bert":                "BERTimbau / BERT-base",
    "logistic_regression": "LR + TF-IDF",
    "lr":                  "LR + TF-IDF",
    "mlp":                 "MLP + TF-IDF",
    "bilstm":              "BiLSTM",
}
_DATASET_DISPLAY: Dict[str, str] = {
    "faketruebr":             "FakeTrue.Br",
    "fakebr":                 "Fake.br-Corpus",
    "fake_br_corpus":         "Fake.br-Corpus",
    "fakerecogna":            "FakeRecogna",
    "fakenewsnet":            "FakeNewsNet",
    "constraint_covid19":     "CONSTRAINT",
    "pheme":                  "PHEME",
}
_MODEL_ORDER: List[str] = [
    "BERTimbau / BERT-base",
    "LR + TF-IDF",
    "MLP + TF-IDF",
    "BiLSTM",
]
_MODEL_COLORS: Dict[str, str] = {
    "BERTimbau / BERT-base": "#1f77b4",
    "LR + TF-IDF":           "#ff7f0e",
    "MLP + TF-IDF":          "#2ca02c",
    "BiLSTM":                "#d62728",
}
_MODEL_LINESTYLES: Dict[str, str] = {
    "BERTimbau / BERT-base": "-",
    "LR + TF-IDF":           "--",
    "MLP + TF-IDF":          "-.",
    "BiLSTM":                ":",
}

# Helpers
def log_counts(df, dataset, stage):
    logger.info(
        "%s | %s | total=%d | fake=%d | real=%d",
        dataset,
        stage,
        len(df),
        int((df["label"] == 1).sum()),
        int((df["label"] == 0).sum()),
    )

def get_combined_dataset_names(
    available_dataset_names: Set[str],
) -> List[str]:
    """
    Retorna os nomes dos datasets combinados gerados pelo EXP_C.
    """
    combined_names: List[str] = []
    combined_groups = {
        "pt": PT_DATASETS,
        "en": EN_DATASETS,
    }
    for lang, dataset_group in combined_groups.items():
        available = sorted(
            available_dataset_names & dataset_group
        )
        if len(available) < 2:
            continue
        for ds_name in available:
            combined_names.append(
                f"combined_{lang}-{ds_name}"
            )
    return combined_names

def preprocess(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    model_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Higieniza o texto PRESERVANDO a caixa (cased) no DataFrame compartilhado.
    A política de caixa é POR MODELO e aplicada depois, em
    run_experiment() → _clean_series(): minúsculas para LR/MLP/BiLSTM
    (se preprocessing.lowercase=true) e texto intacto para os Transformers
    (BERTimbau é cased). Assim o BERTimbau nunca recebe texto em minúsculas.
    """
    if "timestamp" in df.columns:
        raise ValueError(
            "preprocess() deve ser chamado antes da criação da coluna 'timestamp'."
        )
    pp = cfg.get("preprocessing", {})
    df = df.copy()
   
    df["text_raw"] = df["text"]
    df["text"] = df["text"].apply(
        lambda t: clean_text(
            t,
            lowercase=False,
            model_type=model_type,
            remove_url=pp.get("remove_urls", True),
            remove_mention=pp.get("remove_mentions", True),
            hashtag_mode=("normalize" if pp.get("normalize_hashtags", True) else "keep"),
            emoji_mode="keep",
        )
    )
    return df
    
def _build_active_datasets(
    only_dataset: Optional[str],
) -> Dict[str, str]:
    if only_dataset:
        if only_dataset not in DATASET_MAP:
            raise ValueError(
                f"Unknown dataset "
                f"'{only_dataset}'. "
                f"Available: {list(DATASET_MAP)}"
            )
        return {
            only_dataset: (
                DATASET_MAP[only_dataset]
            )
        }
    return DATASET_MAP

def _safe_rglob(
    root: Path,
    max_depth: int = 10,
) -> List[Path]:
    """
    Percorre recursivamente um diretório até max_depth níveis, ignorando symlinks para evitar loops circulares.
    """
    found: List[Path] = []
    def _recurse(
        current: Path,
        depth: int,
    ) -> None:
        if depth > max_depth:
            return
        try:
            for entry in current.iterdir():
                try:
                    if os.path.islink(str(entry)):
                        logger.debug(
                            "_safe_rglob: "
                            "ignorando symlink %s",
                            entry,
                        )
                        continue
                    if entry.is_file():
                        found.append(entry)
                    elif entry.is_dir():
                        _recurse(
                            entry, depth + 1
                        )
                except (
                    PermissionError,
                    OSError,
                    FileNotFoundError,
                ) as e:
                    logger.debug(
                        "_safe_rglob: pulando "
                        "%s - %s",
                        entry, e,
                    )
        except (
            PermissionError,
            OSError,
            FileNotFoundError,
        ) as e:
            logger.debug(
                "_safe_rglob: não listou "
                "%s - %s",
                current, e,
            )
    _recurse(root, depth=0)
    return found

def _hash_dataset_files(
    ds_name: str,
    ds_path_obj: Path,
) -> Dict[str, str]:
    """
    Calcula SHA-256 de arquivos de dados. Limita a _MAX_FILES_TO_HASH arquivos para
    evitar hashing excessivo em datasets com muitos arquivos (ex: PHEME com 102461 arquivos).
    Prioridade:
    1. Arquivos tabulares (.csv, .parquet, .xlsx)
    2. Amostra de JSONs se não houver tabulares
    """
    all_files = _safe_rglob(ds_path_obj)
    tabular_files = [
        f for f in all_files
        if f.suffix.lower() in _TABULAR_EXTENSIONS
    ]
    if tabular_files:
        files_to_hash = (
            tabular_files[:_MAX_FILES_TO_HASH]
        )
    else:
        json_files = [
            f for f in all_files
            if f.suffix.lower()
            in {".json", ".jsonl"}
        ]
        n = len(json_files)
        if n > 0:
            indices = sorted(set([
                0,
                n // 4,
                n // 2,
                3 * n // 4,
                n - 1,
            ]))
            sample = [
                json_files[i]
                for i in indices
                if i < n
            ]
        else:
            sample = []
        files_to_hash = (
            sample[:_MAX_FILES_TO_HASH]
        )
    hashes: Dict[str, str] = {}
    for f in files_to_hash:
        try:
            hashes[f.name] = file_hash(str(f))
        except MemoryError:
            logger.warning(
                "%s | file_hash: arquivo muito "
                "grande: %s (%.1f MB)",
                ds_name, f.name,
                f.stat().st_size / 1e6,
            )
        except Exception as e:
            logger.debug(
                "%s | file_hash: erro em "
                "%s: %s",
                ds_name, f.name, e,
            )
    limited = (
        len(tabular_files) > _MAX_FILES_TO_HASH
        or len(files_to_hash) < len(all_files)
    )
    if hashes:
        logger.info(
            "%s | %d arquivo(s) hasheados "
            "(SHA-256)%s",
            ds_name, len(hashes),
            " [limitado]" if limited else "",
        )
    return hashes

def _apply_sklearn_best_params(
    model_cfg: Dict[str, Any],
    best_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Aplica best_params do GridSearch ao model_cfg.
    Remove os prefixos 'clf__' e 'classifier__' das chaves.
    """
    updated = model_cfg.copy()
    for k, v in best_params.items():
        key = k.replace("clf__", "").replace("classifier__", "")
        updated[key] = v
    return updated

def _build_model_cfg(
    cfg: Dict[str, Any],
    model_name: str,
    ds_name: str,
    mlp_best_params: Dict[str, Dict[str, Any]],
    lr_best_params: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
) -> Dict[str, Any]:
    """
    Monta o model_cfg de (modelo, dataset) aplicando os
    best_params obtidos UMA vez (seed de tuning) para
    MLP e LR. Usado nos EXP_A e EXP_D, de modo que o
    GridSearch não seja refeito a cada seed.
    """
    model_cfg = cfg.get(
        model_name.replace("-", "_"), {}
    ).copy()
    if (
        model_name == "mlp"
        and mlp_best_params.get(ds_name)
    ):
        model_cfg = _apply_sklearn_best_params(
            model_cfg, mlp_best_params[ds_name]
        )
        model_cfg["skip_grid_search"] = True
    if (
        model_name == "logistic_regression"
        and lr_best_params is not None
        and lr_best_params.get(ds_name)
    ):
        model_cfg = _apply_sklearn_best_params(
            model_cfg, lr_best_params[ds_name]
        )
        model_cfg.pop("param_grid", None)
        model_cfg["skip_grid_search"] = True
    return model_cfg

def _log_run_end(_run_start: datetime ) -> None:
    _run_end = datetime.now()
    elapsed  = _run_end - _run_start
    logger.info("=" * 70)
    logger.info(
        "RUN END - %s | elapsed=%s",
        _run_end.isoformat(), elapsed,
    )
    logger.info("=" * 70)

# Hyperparameter tuning - sklearn (LR + MLP)
def _tune_sklearn_once(
    raw: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
    tuning_seed: int = 42,
    model_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Executa GridSearchCV para modelos sklearn (LR e MLP) UMA vez por dataset.
    Retorna:
    Dict[model_name, Dict[ds_name, best_params]]
    """
    if model_names is None:
        model_names = [
            "logistic_regression", "mlp"
        ]
    results: Dict[str, Dict[str, Any]] = {
        m: {} for m in model_names
    }
    sc = cfg.get("splitting", {})
    logger.info("=" * 70)
    logger.info(
        "SKLEARN HYPERPARAMETER TUNING - "
        "tuning_seed=%d | modelos=%s",
        tuning_seed, model_names,
    )
    logger.info("=" * 70)
    for model_name in model_names:
        model_cfg  = cfg.get(
            model_name.replace("-", "_"), {}
        ).copy()
        param_grid = model_cfg.pop(
            "param_grid", None
        )
        if not param_grid:
            logger.info(
                "%s GridSearch disabled: "
                "no param_grid.",
                model_name,
            )
            continue
        for ds_name, df in raw.items():
            logger.info(
                "%s GridSearch | dataset=%s",
                model_name, ds_name,
            )
            _grp = "_nd_group" if "_nd_group" in df.columns else None
            train_df, val_df, _ = random_split(
                df,
                test_size=sc.get(
                    "test_size", 0.20
                ),
                val_size=sc.get(
                    "validation_size", 0.10
                ),
                seed=tuning_seed,
                group_col=_grp,
            )
            for _sdf in (train_df, val_df):
                _sdf.drop(columns=["_nd_group"], errors="ignore", inplace=True)
            try:
                result = run_experiment(
                    experiment_id=(
                        f"{model_name.upper()}"
                        f"_TUNING_{ds_name}"
                        f"_seed{tuning_seed}"
                    ),
                    model_name=model_name,
                    dataset_name=ds_name,
                    train_df=train_df,
                    val_df=val_df,
                    # test_df=val_df é intencional: durante o tuning não há conjunto de teste independente.
                    # As métricas de "teste" aqui são descartadas, apenas best_params é aproveitado.
                    test_df=val_df,
                    cfg=cfg,
                    seed=tuning_seed,
                    model_kwargs={
                        **model_cfg,
                        "param_grid": param_grid,
                    },
                    save_model=False,
                    save_metrics_flag=True,
                    save_predictions_flag=False,
                )
                best = result.get("best_params")
                if best:
                    results[model_name][
                        ds_name
                    ] = best
                    logger.info(
                        "%s GridSearch | %s -> "
                        "best_params=%s",
                        model_name, ds_name, best,
                    )
                else:
                    results[model_name][
                        ds_name
                    ] = {}
                    logger.warning(
                        "%s GridSearch | %s -> "
                        "best_params não "
                        "retornados. "
                        "Usando defaults.",
                        model_name, ds_name,
                    )
            except Exception as e:
                logger.warning(
                    "%s GridSearch falhou para "
                    "%s: %s - usando defaults.",
                    model_name, ds_name, e,
                )
                results[model_name][ds_name] = {}
    return results

def _load_or_tune_sklearn(
    raw: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
    tuning_seed: int,
    model_names: List[str],
    metrics_dir: str = "results/metrics",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Tenta carregar best_params do disco.
    Se não encontrar, executa tuning.
    Evita re-execução de GridSearch quando os resultados já existem (~90 min desnecessários).
    """
    results: Dict[
        str, Dict[str, Dict[str, Any]]
    ] = {m: {} for m in model_names}
    metrics_path = Path(metrics_dir)
    for model_name in model_names:
        loaded_count = 0
        for ds_name in raw.keys():
            bp = _extract_best_params(
                model_name=model_name,
                ds_name=ds_name,
                tuning_seed=tuning_seed,
                metrics_path=metrics_path,
            )
            if bp:
                results[model_name][
                    ds_name
                ] = bp
                loaded_count += 1
                logger.info(
                    "%s: best_params "
                    "carregados para %s: %s",
                    model_name, ds_name, bp,
                )
            else:
                logger.debug(
                    "%s: best_params não "
                    "encontrado para %s",
                    model_name, ds_name,
                )
        total_ds = len(raw)
        if loaded_count == total_ds:
            logger.info(
                "%s: todos best_params "
                "do disco (%d/%d) - "
                "GridSearch pulado.",
                model_name,
                loaded_count, total_ds,
            )
        elif loaded_count > 0:
            missing_ds = [
                ds for ds in raw.keys()
                if ds not in
                results[model_name]
            ]
            logger.warning(
                "%s: %d/%d carregados. "
                "Tunando restantes: %s",
                model_name,
                loaded_count, total_ds,
                missing_ds,
            )
            partial = _tune_sklearn_once(
                raw={
                    k: v
                    for k, v in raw.items()
                    if k in missing_ds
                },
                cfg=cfg,
                tuning_seed=tuning_seed,
                model_names=[model_name],
            )
            for ds, bp in (
                partial.get(
                    model_name, {}
                ).items()
            ):
                results[model_name][ds] = bp
        else:
            logger.info(
                "%s: nenhum best_params "
                "no disco - GridSearch "
                "completo.",
                model_name,
            )
            full = _tune_sklearn_once(
                raw=raw,
                cfg=cfg,
                tuning_seed=tuning_seed,
                model_names=[model_name],
            )
            results[model_name] = (
                full.get(model_name, {})
            )
    return results

def _extract_best_params(
    model_name: str,
    ds_name: str,
    tuning_seed: int,
    metrics_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Extrai best_params de fontes JSON (sem pickle, que executa código arbitrário ao desserializar):
    1. JSON EXP_A -> campo "best_params"
    2. JSON EXP_A -> campo "grid_meta"
    3. JSON TUNING do STEP 3A
    Os hiperparâmetros são selecionados UMA vez (tuning_seed) e reutilizados nas demais seeds.
    """
    # Fonte 1 e 2 - JSON EXP_A
    json_path = (
        metrics_path
        / f"EXP_A_{model_name}"
          f"_{ds_name}"
          f"_seed{tuning_seed}.json"
    )
    if json_path.exists():
        try:
            data = json.loads(
                json_path.read_text(
                    encoding="utf-8"
                )
            )
            bp = data.get("best_params")
            if bp:
                return bp
            gm = data.get("grid_meta", {})
            bp = gm.get("best_params")
            if bp:
                return bp
        except (OSError, ValueError, AttributeError) as e:
            logger.warning(
                "_extract_best_params: "
                "erro ao ler %s: %s",
                json_path.name, e,
            )
    # Fonte 3 - JSON TUNING
    tuning_path = (
        metrics_path
        / f"{model_name.upper()}"
          f"_TUNING_{ds_name}"
          f"_seed{tuning_seed}.json"
    )
    if tuning_path.exists():
        try:
            data = json.loads(
                tuning_path.read_text(
                    encoding="utf-8"
                )
            )
            bp = data.get("best_params")
            if bp:
                logger.info(
                    "_extract_best_params: "
                    "best_params do TUNING "
                    "%s: %s",
                    tuning_path.name, bp,
                )
                return bp
        except (OSError, ValueError, AttributeError) as e:
            logger.warning(
                "_extract_best_params: "
                "erro ao ler TUNING "
                "%s: %s",
                tuning_path.name, e,
            )
    return None

def _diagnose_duplicates(
    df: pd.DataFrame,
    ds_name: str,
) -> None:
    """
    Registra estatísticas de duplicatas para inspeção manual antes da deduplicação.
    Ativado via cfg['debug']['diagnose_duplicates'].
    """
    df = df.copy()
    df["_text_hash_key"] = (
        df["text"].fillna("").astype(str)
        .str.lower().str.strip()
    )
    duplicados = df[
        df.duplicated("_text_hash_key", keep=False)
    ].copy()
    logger.info(
        "%s | total=%d | em grupos duplicados=%d"
        " | textos distintos duplicados=%d",
        ds_name, len(df), len(duplicados),
        duplicados["_text_hash_key"].nunique(),
    )
    rotulos = duplicados.groupby(
        "_text_hash_key"
    )["label"].nunique()
    conflitos = rotulos[rotulos > 1]
    logger.info(
        "%s | grupos com conflito de rótulo: %d",
        ds_name, len(conflitos),
    )
    if len(conflitos) > 0:
        exemplos = duplicados[
            duplicados["_text_hash_key"]
            .isin(conflitos.index)
        ][["text", "label", "dataset"]]
        logger.info(
            "%s | exemplos conflitantes:\n%s",
            ds_name,
            exemplos.head(10).to_string(index=False),
        )
        
# Single-seed runner
def _run_single_seed(
    seed: int,
    raw: Dict[str, pd.DataFrame],
    model_names: List[str],
    cfg: Dict[str, Any],
    mlp_best_params: Dict[str, Dict[str, Any]],
    lr_best_params: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
    active_seeds: Optional[List[int]] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Executa todos os experimentos (A–D) para uma seed.
    Parâmetros
    ----------
    seed            : seed da execução.
    raw             : datasets carregados.
    model_names     : modelos ativos.
    cfg             : configuração.
    mlp_best_params : best_params MLP por dataset.
    lr_best_params  : best_params LR por dataset.
    active_seeds    : todas as seeds da run
                      (usado para log EXP_D).
    Retorna
    -------
    Dict {(model_name, ds_name): f1_macro}
    """
    set_seed(seed)
    logger.info("=" * 70)
    logger.info(
        "SEED %d - starting full experiment run",
        seed,
    )
    logger.info("=" * 70)
    sc = cfg.get("splitting", {})
    first_seed = (
        active_seeds[0]
        if active_seeds is not None
        else seed
    )
    # Splits 
    splits: Dict[
        str, Dict[str, pd.DataFrame]
    ] = {}
    for ds_name, df in raw.items():
        group_col = "_nd_group" if "_nd_group" in df.columns else None
        train, val, test = random_split(
            df,
            test_size=sc.get("test_size", 0.20),
            val_size=sc.get(
                "validation_size", 0.10
            ),
            seed=seed,
            group_col=group_col,
        )
        # Remove internal grouping column before it reaches any model
        for split_df in (train, val, test):
            split_df.drop(columns=["_nd_group"], errors="ignore", inplace=True)
        splits[ds_name] = {
            "train": train,
            "val":   val,
            "test":  test,
        }
        logger.info(
            "Seed %d | Split %s: "
            "train=%d val=%d test=%d",
            seed, ds_name,
            len(train), len(val), len(test),
        )
    # EXPERIMENT A 
    logger.info(
        "--- Seed %d | EXPERIMENT A ---", seed
    )
    intra_f1: Dict[
        Tuple[str, str], float
    ] = {}
    for ds_name in raw:
        sp = splits[ds_name]
        for model_name in model_names:
            exp_id = (
                f"EXP_A_{model_name}"
                f"_{ds_name}_seed{seed}"
            )
            model_cfg = _build_model_cfg(
                cfg, model_name, ds_name,
                mlp_best_params, lr_best_params,
            )
            try:
                result = run_experiment(
                    experiment_id=exp_id,
                    model_name=model_name,
                    dataset_name=ds_name,
                    train_df=sp["train"],
                    val_df=sp["val"],
                    test_df=sp["test"],
                    cfg=cfg,
                    seed=seed,
                    model_kwargs=model_cfg,
                )
                m = result.get("metrics", {})
                f1 = m.get("f1_macro", float("nan"))
                intra_f1[
                    (model_name, ds_name)
                ] = float(f1)
                
                imbalanced_ds = cfg.get(
                    "evaluation", {}
                ).get("imbalanced_datasets", [])
                ratio_threshold = float(
                    cfg.get("evaluation", {})
                    .get("imbalance_ratio_threshold", 5)
                )
                test_counts = (
                    sp["test"]["label"].value_counts()
                )
                ratio = (
                    test_counts.max() / test_counts.min()
                    if len(test_counts) >= 2
                    else float("inf")
                )
                if (
                    ds_name in imbalanced_ds
                    or ratio >= ratio_threshold
                ):
                    f1w  = m.get("f1_weighted", float("nan"))
                    prauc = m.get("pr_auc", float("nan"))
                    logger.warning(
                        "EXP_A | %s | %s | seed=%d: "
                        "dataset desbalanceado (ratio=%.1f:1) - "
                        "f1_macro=%.4f pode ser enganoso; "
                        "f1_weighted=%.4f, pr_auc=%.4f "
                        "(métricas primárias para este dataset).",
                        model_name, ds_name, seed,
                        ratio, f1, f1w, prauc,
                    )
            except Exception as e:
                logger.error(
                    "EXP_A | %s | %s | "
                    "seed=%d: %s",
                    model_name, ds_name, seed, e,
                )
    # EXPERIMENT B 
    logger.info(
        "--- Seed %d | EXPERIMENT B ---", seed
    )
    try:
        run_all_cross_dataset(
            datasets=splits,
            model_names=model_names,
            model_root="results/models",
            cfg=cfg,
            seed=seed,
        )
    except Exception as e:
        logger.error(
            "EXP_B | seed=%d: %s", seed, e
        )
    # EXPERIMENT C 
    logger.info(
        "--- Seed %d | EXPERIMENT C ---", seed
    )
    _run_experiment_c(
        seed=seed,
        splits=splits,
        raw=raw,
        model_names=model_names,
        cfg=cfg,
        mlp_best_params=mlp_best_params,
        lr_best_params=lr_best_params,
    )
    # EXPERIMENT D 
    logger.info(
        "--- Seed %d | EXPERIMENT D ---", seed
    )
    for ds_name, df in raw.items():
        if "timestamp" not in df.columns:
            logger.info(
                "Seed %d | Skipping %s "
                "(no timestamp)",
                seed, ds_name,
            )
            continue
        if ds_name not in TEMPORAL_DATASETS:
            logger.info(
                "Seed %d | Skipping %s "
                "(not in TEMPORAL_DATASETS)",
                seed, ds_name,
            )
            continue
        if df["timestamp"].isna().all():
            logger.warning(
                "Seed %d | Skipping %s "
                "(all timestamps NaT)",
                seed, ds_name,
            )
            continue
        n_total     = len(df)
        df_temporal = df[
            df["timestamp"].notna()
        ].reset_index(drop=True)
        n_valid = len(df_temporal)
        if n_valid < n_total:
            logger.warning(
                "Seed %d | %s: %d/%d exemplos "
                "sem timestamp excluídos do EXP_D.",
                seed, ds_name,
                n_total - n_valid, n_total,
            )
        if n_valid < 50:
            logger.warning(
                "Seed %d | Skipping %s "
                "(apenas %d exemplos com timestamp "
                "- mínimo 50)",
                seed, ds_name, n_valid,
            )
            continue
        logger.info(
            "Seed %d | EXPERIMENT D | %s | "
            "timestamp válido: %d/%d",
            seed, ds_name, n_valid, n_total,
        )
        # Aviso antecipado para datasets fortemente desbalanceados: o split temporal pode falhar
        # com ValueError se a classe minoritária tiver < 5 amostras em alguma partição.
        # PHEME (ratio ~366:1) é o caso típico. 
        # O erro é esperado e tratado abaixo: o dataset é pulado com WARNING.
        if "label" in df_temporal.columns:
            _label_ratio = (
                df_temporal["label"].value_counts()
            )
            if len(_label_ratio) >= 2:
                _ratio = (
                    _label_ratio.max()
                    / _label_ratio.min()
                )
                if _ratio > 50:
                    logger.warning(
                        "Seed %d | EXP_D | %s: "
                        "dataset fortemente desbalanceado "
                        "(ratio=%.0f:1) - split temporal "
                        "pode ser inviável. "
                        "Se ValueError ocorrer, o dataset "
                        "será pulado (comportamento "
                        "esperado, documentado na "
                        "dissertação).",
                        seed, ds_name, _ratio,
                    )
        if seed == first_seed:
            logger.info(
                "Seed %d | EXP_D | %s | "
                "NOTA: split temporal é "
                "determinístico - mesmos dados "
                "para todas as seeds. "
                "Variação entre seeds reflete "
                "apenas inicialização do modelo.",
                seed, ds_name,
            )
        for model_name in model_names:
            logger.info(
                "Seed %d | EXPERIMENT D | "
                "%s | %s",
                seed, ds_name, model_name,
            )
            try:
                run_temporal_experiment(
                    dataset_name=ds_name,
                    df=df_temporal,
                    model_name=model_name,
                    cfg=cfg,
                    seed=seed,
                    model_kwargs=_build_model_cfg(
                        cfg, model_name, ds_name,
                        mlp_best_params,
                        lr_best_params,
                    ),
                )
            except ValueError as e:
                # Split temporal inválido (ex.: classe minoritária insuficiente após corte cronológico - comum em datasets
                # fortemente desbalanceados como PHEME).
                # Registrado como WARNING, não ERROR: o comportamento é esperado e documentado na dissertação (Seção EXP_D).
                logger.warning(
                    "EXP_D | %s | %s | "
                    "seed=%d: %s - pulando.",
                    ds_name, model_name,
                    seed, e,
                )
            except Exception as e:
                logger.error(
                    "EXP_D | %s | %s | "
                    "seed=%d: %s",
                    ds_name, model_name,
                    seed, e,
                )
    
    logger.info("SEED %d - complete", seed)
    return intra_f1

# Experiment C
def _run_experiment_c(
    seed: int,
    splits: Dict[str, Dict[str, pd.DataFrame]],
    raw: Dict[str, pd.DataFrame],
    model_names: List[str],
    cfg: Dict[str, Any],
    mlp_best_params: Dict[str, Dict[str, Any]],
    lr_best_params: Optional[
        Dict[str, Dict[str, Any]]
    ] = None,
) -> None:
    """
    Experimento C - Generalização Cross-Corpus.
    O modelo é treinado UMA vez no pool combinado e testado em cada dataset individualmente via skip_training=True.
    """
    combined_groups = {
        "pt": sorted(
            set(raw.keys()) & PT_DATASETS
        ),
        "en": sorted(
            set(raw.keys()) & EN_DATASETS
        ),
    }
    for lang, available in (
        combined_groups.items()
    ):
        if len(available) < 2:
            logger.info(
                "[EXP_C_%s] Apenas %d "
                "dataset(s) - pulando "
                "(mínimo 2).",
                lang.upper(), len(available),
            )
            continue
        logger.info(
            "[EXP_C_%s] Pool combinado: %s",
            lang.upper(), available,
        )
        # Remover colunas temporais
        train_frames: List[pd.DataFrame] = []
        val_frames:   List[pd.DataFrame] = []
        for ds_name in available:
            sp         = splits[ds_name]
            train_part = sp["train"].copy()
            val_part   = sp["val"].copy()
            for col in (
                "timestamp", "event", "veracity"
            ):
                train_part = train_part.drop(
                    columns=[col], errors="ignore"
                )
                val_part = val_part.drop(
                    columns=[col], errors="ignore"
                )
            train_frames.append(train_part)
            val_frames.append(val_part)
        combined_train = pd.concat(
            train_frames, ignore_index=True
        )
        combined_val = pd.concat(
            val_frames, ignore_index=True
        )
        combined_train = remove_exact_duplicates(
            combined_train
        )
        combined_val = remove_exact_duplicates(
            combined_val
        )
        logger.info(
            "[EXP_C_%s] train=%d val=%d",
            lang.upper(),
            len(combined_train),
            len(combined_val),
        )
        # Hashes do pool de treino+validação: qualquer
        # texto de teste idêntico a eles é removido do teste (vazamento entre corpora do pool).
        pool_hashes = set(
            combined_train["text"].fillna("").map(text_hash)
        ) | set(
            combined_val["text"].fillna("").map(text_hash)
        )
        for model_name in model_names:
            model_cfg = cfg.get(
                model_name.replace("-", "_"), {}
            ).copy()
            for key in _ALWAYS_REMOVE_KEYS:
                model_cfg.pop(key, None)
            if model_name != "bilstm":
                for key in _BILSTM_ONLY_KEYS:
                    model_cfg.pop(key, None)

            # MLP best_params: usa-se o ajuste do primeiro
            # dataset do pool (ordem alfabética). O pool não
            # possui GridSearch próprio para evitar que o
            # conjunto de teste de qualquer corpus influencie
            # a seleção de hiperparâmetros.
            if model_name == "mlp":
                first_ds = available[0]
                if (
                    first_ds in mlp_best_params
                    and mlp_best_params[first_ds]
                ):
                    model_cfg = (
                        _apply_sklearn_best_params(
                            model_cfg,
                            mlp_best_params[first_ds],
                        )
                    )
                    model_cfg[
                        "skip_grid_search"
                    ] = True
            # LR best_params
            if model_name == "logistic_regression":
                bp: Optional[Dict[str, Any]] = None
                if lr_best_params is not None:
                    for ds in available:
                        if (
                            ds in lr_best_params
                            and lr_best_params[ds]
                        ):
                            bp = lr_best_params[ds]
                            logger.info(
                                "[EXP_C_%s] LR: "
                                "best_params de "
                                "%s: %s",
                                lang.upper(), ds, bp,
                            )
                            break
                if bp is None:
                    for ds in available:
                        bp_path = Path(
                            "results/metrics/"
                            f"EXP_A_logistic"
                            f"_regression"
                            f"_{ds}"
                            f"_seed{seed}.json"
                        )
                        if bp_path.exists():
                            try:
                                data = json.loads(
                                    bp_path.read_text(
                                        encoding="utf-8"
                                    )
                                )
                                bp = data.get(
                                    "best_params"
                                )
                                if bp:
                                    logger.info(
                                        "[EXP_C_%s] "
                                        "LR: best_params"
                                        " de %s "
                                        "(JSON): %s",
                                        lang.upper(),
                                        ds, bp,
                                    )
                                    break
                            except Exception:
                                pass
                if bp:
                    model_cfg = (
                        _apply_sklearn_best_params(
                            model_cfg, bp
                        )
                    )
                    model_cfg.pop(
                        "param_grid", None
                    )
                    model_cfg[
                        "skip_grid_search"
                    ] = True
                else:
                    logger.warning(
                        "[EXP_C_%s] LR: "
                        "best_params não "
                        "encontrados. GridSearch "
                        "será executado no pool "
                        "combinado.",
                        lang.upper(),
                    )
            # Treinar UMA vez 
            train_exp_id = (
                f"EXP_C_{lang}_{model_name}"
                f"_combined_{lang}"
                f"_TRAIN_seed{seed}"
            )
            logger.info(
                "[EXP_C_%s] Treinando %s "
                "(seed=%d)...",
                lang.upper(), model_name, seed,
            )
            try:
                train_result = run_experiment(
                    experiment_id=train_exp_id,
                    model_name=model_name,
                    dataset_name=(
                        f"combined_{lang}"
                    ),
                    train_df=combined_train,
                    val_df=combined_val,
                    test_df=combined_val,
                    cfg=cfg,
                    seed=seed,
                    model_kwargs=model_cfg,
                    language=lang,
                    save_model=True,
                    save_metrics_flag=False,
                    save_predictions_flag=False,
                )
                trained_model = (
                    train_result.get("model")
                )
                if trained_model is None:
                    logger.error(
                        "[EXP_C_%s] %s: modelo "
                        "não retornado - "
                        "pulando testes.",
                        lang.upper(), model_name,
                    )
                    continue
                logger.info(
                    "[EXP_C_%s] %s treinado OK"
                    " - testando em %d "
                    "dataset(s)...",
                    lang.upper(), model_name,
                    len(available),
                )
            except Exception as e:
                logger.error(
                    "[EXP_C_%s] Treino %s "
                    "falhou: %s - pulando.",
                    lang.upper(), model_name, e,
                )
                continue
            # Testar em cada dataset 
            for ds_name in available:
                test_exp_id = (
                    f"EXP_C_{lang}_{model_name}"
                    f"_combined_{lang}"
                    f"_test_{ds_name}_seed{seed}"
                )
                test_df = (
                    splits[ds_name]["test"].copy()
                )
                overlap = (
                    test_df["text"].fillna("")
                    .map(text_hash)
                    .isin(pool_hashes)
                )
                if overlap.any():
                    logger.warning(
                        "[EXP_C_%s] %s: %d/%d textos "
                        "de teste presentes no pool "
                        "de treino/val - removidos "
                        "(vazamento).",
                        lang.upper(), ds_name,
                        int(overlap.sum()), len(test_df),
                    )
                    test_df = test_df[~overlap]
                if test_df["label"].nunique() < 2:
                    logger.warning(
                        "[EXP_C_%s] %s: teste sem as "
                        "duas classes após filtro - "
                        "pulando.",
                        lang.upper(), ds_name,
                    )
                    continue
                for col in (
                    "timestamp", "event", "veracity"
                ):
                    test_df = test_df.drop(
                        columns=[col],
                        errors="ignore",
                    )
                logger.info(
                    "[EXP_C_%s] Testando %s "
                    "em %s...",
                    lang.upper(),
                    model_name, ds_name,
                )
                try:
                    run_experiment(
                        experiment_id=test_exp_id,
                        model_name=model_name,
                        dataset_name=(
                            f"combined_{lang}"
                            f"-{ds_name}"
                        ),
                        train_df=combined_train,
                        val_df=combined_val,
                        test_df=test_df,
                        cfg=cfg,
                        seed=seed,
                        model_kwargs=model_cfg,
                        language=lang,
                        pretrained_model=(
                            trained_model
                        ),
                        skip_training=True,
                        save_model=False,
                        save_metrics_flag=True,
                        save_predictions_flag=True,
                    )
                    logger.info(
                        "[EXP_C_%s] %s -> %s: OK",
                        lang.upper(),
                        model_name, ds_name,
                    )
                except Exception as e:
                    logger.error(
                        "[EXP_C_%s] Teste "
                        "%s->%s falhou: %s",
                        lang.upper(),
                        model_name, ds_name, e,
                    )

# Post-processing helper
def _run_postprocess() -> None:
    """
    Executa scripts de pós-processamento.
    """
    logger.info(
        "Post-processing: "
        "tables, figures, report"
    )
    post_scripts = [
        "scripts/build_summary_csvs.py",
        "scripts/generate_tables.py",
        "scripts/generate_figures.py",
        "scripts/generate_report.py",
    ]
    for script in post_scripts:
        script_path = Path(script)
        if not script_path.exists():
            logger.warning(
                "Script não encontrado: "
                "%s - pulando.",
                script,
            )
            continue
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                ],
                check=True,
                timeout=_POSTPROCESS_TIMEOUT,
            )
            logger.info(
                "Post-processing OK: %s",
                script,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "Post-processing timeout "
                "(%ds): %s - pulando.",
                _POSTPROCESS_TIMEOUT, script,
            )
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Post-processing falhou: "
                "%s - código %d",
                script, e.returncode,
            )
        except Exception as e:
            logger.warning(
                "Post-processing erro: "
                "%s - %s",
                script, e,
            )


# STEP 7 helpers - Text length stats + ROC curves
def _compute_text_length_stats(
    raw: Dict[str, pd.DataFrame],
    output_dir: str = "results",
) -> None:
    """
    Calcula média, mediana e desvio padrão do comprimento dos textos (em palavras) por corpus
    e salva CSV + tabela LaTeX.
    Usa os DataFrames já carregados e pré-processados pelo STEP 1 - não recarrega os datasets.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds_key, df in raw.items():
        display = _DATASET_DISPLAY.get(
            ds_key, ds_key
        )
        idioma = (
            "PT-BR"
            if ds_key in PT_DATASETS
            else "EN"
        )
        if "text" not in df.columns:
            logger.warning(
                "STEP 7 | %s: coluna 'text' "
                "ausente - pulando.",
                ds_key,
            )
            continue
        lengths = (
            df["text"]
            .fillna("")
            .apply(lambda t: len(t.split()))
            .tolist()
        )
        if not lengths:
            logger.warning(
                "STEP 7 | %s: sem textos "
                "válidos - pulando.",
                ds_key,
            )
            continue
        arr = np.array(lengths, dtype=float)
        rows.append({
            "dataset": display,
            "idioma":  idioma,
            "n":       int(len(arr)),
            "mean":    float(np.mean(arr)),
            "median":  float(np.median(arr)),
            "std":     float(np.std(arr, ddof=1)),
            "min":     int(np.min(arr)),
            "max":     int(np.max(arr)),
        })
        logger.info(
            "STEP 7 | %s: n=%d | "
            "média=%.1f | mediana=%.1f | "
            "dp=%.1f | max=%d",
            display,
            int(len(arr)),
            float(np.mean(arr)),
            float(np.median(arr)),
            float(np.std(arr, ddof=1)),
            int(np.max(arr)),
        )
    if not rows:
        logger.warning(
            "STEP 7 | Nenhum dado para "
            "estatísticas de comprimento."
        )
        return
    df_stats = pd.DataFrame(rows)
    # CSV
    csv_path = out / "text_length_stats.csv"
    df_stats.to_csv(csv_path, index=False)
    logger.info(
        "STEP 7 | text_length_stats.csv "
        "salvo: %s",
        csv_path,
    )
    # LaTeX
    tex_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Estatísticas de comprimento "
        r"dos textos por corpus após "
        r"pré-processamento. Comprimento medido "
        r"em palavras (tokens por espaço em "
        r"branco).}",
        r"\label{tab:text-length-stats}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{%",
        r"    >{\raggedright\arraybackslash}"
        r"p{3.0cm}",
        r"    >{\centering\arraybackslash}"
        r"p{1.2cm}",
        r"    >{\centering\arraybackslash}X",
        r"    >{\centering\arraybackslash}X",
        r"    >{\centering\arraybackslash}X",
        r"    >{\centering\arraybackslash}X",
        r"    >{\centering\arraybackslash}X}",
        r"\hline",
        r"\textbf{Dataset} & "
        r"\textbf{Idioma} & "
        r"\textbf{$n$} & "
        r"\textbf{Média} & "
        r"\textbf{Mediana} & "
        r"\textbf{DP} & "
        r"\textbf{Máx.} \\",
        r"\hline",
    ]
    for _, row in df_stats.iterrows():
        line = (
            f"{row['dataset']} & "
            f"{row['idioma']} & " 
            f"{int(row['n']):,} & "
            f"{row['mean']:.1f} & "
            f"{row['median']:.1f} & "
            f"{row['std']:.1f} & "
            f"{int(row['max']):,} \\\\"
        )
        # Separador decimal brasileiro
        line = line.replace(",", "{,}")
        tex_lines.append(line)
    tex_lines.extend([
        r"\hline",
        r"\end{tabularx}",
        r"\begin{flushleft}",
        r"\footnotesize",
        r"Nota: comprimento medido em palavras "
        r"após pré-processamento padrão "
        r"(remoção de URLs, menções, "
        r"normalização Unicode e truncamento "
        r"a 5.000 caracteres). "
        r"DP~=~desvio padrão amostral. "
        r"$n$~=~número de exemplos após "
        r"deduplicação. "
        r"Fonte: elaborado pela autora.",
        r"\end{flushleft}",
        r"\end{table}",
    ])
    tex_path = out / "text_length_stats.tex"
    tex_path.write_text(
        "\n".join(tex_lines),
        encoding="utf-8",
    )
    logger.info(
        "STEP 7 | text_length_stats.tex "
        "salvo: %s",
        tex_path,
    )

def _fmt_row(
    ds_name: str,
    row: pd.Series,
) -> str:
    """
    Formata uma linha da tabela LaTeX de AUC-ROC.
    O maior valor da linha recebe \\textbf{}. Valores ausentes são exibidos como ---. 
    Tolerância 1e-4 é compatível com o arredondamento a 4 casas decimais de auc_mean.
    """
    valid = [
        v for v in row.values
        if not np.isnan(v)
    ]
    max_val = (
        max(valid) if valid else None
    )
    cells = []
    for val in row.values:
        if np.isnan(val):
            cells.append("---")
        else:
            fmt = (
                f"{val:.4f}"
                .replace(".", "{,}")
            )
            if (
                max_val is not None
                and abs(val - max_val) < 1e-4
            ):
                fmt = (
                    r"\textbf{" + fmt + "}"
                )
            cells.append(fmt)
    return (
        str(ds_name)
        + " & "
        + " & ".join(cells)
        + r" \\"
    )

def _parse(
    path: Path,
    seeds: List[int],
) -> Optional[dict]:
    name = path.stem
    if not name.startswith("EXP_A_"):
        return None
    name = name[len("EXP_A_"):]
    if name.endswith("_predictions"):
        name = name[: -len("_predictions")]
    seed_match = _re.search(
        r"_?(seed\d+)$", name
    )
    if not seed_match:
        return None
    seed_val = int(
        seed_match.group(1).replace("seed", "")
    )
    if seed_val not in seeds:
        return None
    name = name[: seed_match.start()].strip("_")
    model_key = None
    for key in sorted(
        _MODEL_DISPLAY.keys(),
        key=len,
        reverse=True,
    ):
        if name.startswith(key):
            model_key = key
            name = name[len(key):].strip("_")
            break
    if model_key is None:
        return None
    return {
        "model_key":   model_key,
        "dataset_key": name,
        "seed":        seed_val,
        "path":        path,
    }

# Carregamento 
def _load(path: Path) -> Optional[dict]:
    try:
        with open(
            path, encoding="utf-8"
        ) as f:
            data = json.load(f)
    except Exception as e:
        logger.error(
            "STEP 7 | ROC: erro ao "
            "carregar %s: %s",
            path.name, e,
        )
        return None
    if "y_true" not in data:
        logger.error(
            "STEP 7 | ROC: 'y_true' "
            "ausente em %s.",
            path.name,
        )
        return None
    raw_proba = data.get(
        "y_proba",
        data.get("y_scores"),
    )
    if raw_proba is None:
        logger.error(
            "STEP 7 | ROC: 'y_proba' "
            "ausente em %s.",
            path.name,
        )
        return None
    y_true  = np.array(data["y_true"])
    y_proba = np.array(raw_proba)
    if y_proba.ndim == 2:
        y_scores = y_proba[:, 1]
    elif y_proba.ndim == 1:
        y_scores = y_proba
    else:
        logger.error(
            "STEP 7 | ROC: shape "
            "inesperado %s em %s.",
            y_proba.shape, path.name,
        )
        return None
    return {
        "y_true":   y_true,
        "y_scores": y_scores,
    }

# Curva ROC média por dataset 
def _mean_roc(
    roc_list: list,
) -> tuple:
    base_fpr = np.linspace(0, 1, 200)
    tprs     = []
    aucs     = []
    for fpr, tpr, auc_val in roc_list:
        tpr_interp = np.interp(
            base_fpr, fpr, tpr
        )
        tpr_interp[0]  = 0.0
        tpr_interp[-1] = 1.0
        tprs.append(tpr_interp)
        aucs.append(auc_val)
    tprs     = np.array(tprs)
    tpr_mean = tprs.mean(axis=0)

    if tprs.shape[0] > 1:
        tpr_std  = tprs.std(axis=0, ddof=1)
        auc_std  = float(np.std(aucs, ddof=1))
    else:
        tpr_std  = np.zeros_like(tpr_mean)
        auc_std  = 0.0
    auc_mean = float(np.mean(aucs))
    return (
        base_fpr,
        tpr_mean,
        tpr_std,
        auc_mean,
        auc_std,
    )

def _generate_roc_curves(
    seeds: List[int],
    model_names: List[str],
    predictions_dir: str = "results/predictions",
    output_dir: str = "results/figures",
) -> None:
    """
    Gera curvas ROC médias por dataset a partir dos arquivos de predição do EXP_A.
    Espera arquivos no padrão:
        EXP_A_{model}_{dataset}_seed{N}_predictions.json
    Cada arquivo deve conter:
        y_true  : lista de rótulos verdadeiros
        y_proba : lista [[p0, p1], ...] ou [p1, ...]
    """
    pred_path = Path(predictions_dir)
    out_path  = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pred_files = sorted(
        pred_path.glob(
            "EXP_A_*_predictions.json"
        )
    )
    if not pred_files:
        logger.warning(
            "STEP 7 | Nenhum arquivo "
            "EXP_A_*_predictions.json em %s "
            "- curvas ROC não geradas.",
            pred_path,
        )
        return
    logger.info(
        "STEP 7 | ROC: %d arquivo(s) "
        "encontrado(s).",
        len(pred_files),
    )
    
    # Coleta dados por dataset/modelo 
    # data[dataset_display][model_display]
    #     = [(fpr, tpr, auc_val), ...]
    data: Dict[str, Dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in pred_files:
        parsed = _parse(path, seeds)
        if parsed is None:
            continue
        preds = _load(path)
        if preds is None:
            continue
        y_true   = preds["y_true"]
        y_scores = preds["y_scores"]
        if len(np.unique(y_true)) < 2:
            logger.warning(
                "STEP 7 | ROC: classe única "
                "em %s - pulando.",
                path.name,
            )
            continue
        try:
            fpr, tpr, _ = roc_curve(
                y_true,
                y_scores,
                pos_label=1,
            )
            auc_val = auc(fpr, tpr)
        except Exception as e:
            logger.error(
                "STEP 7 | ROC: erro ao "
                "calcular curva para %s: %s",
                path.name, e,
            )
            continue
        model_display = _MODEL_DISPLAY.get(
            parsed["model_key"],
            parsed["model_key"],
        )
        dataset_display = _DATASET_DISPLAY.get(
            parsed["dataset_key"],
            parsed["dataset_key"],
        )
        data[dataset_display][
            model_display
        ].append((fpr, tpr, auc_val))
    if not data:
        logger.warning(
            "STEP 7 | ROC: nenhum dado válido "
            "carregado - curvas ROC não geradas."
        )
        return
    
    # Geração das figuras 
    summary_rows = []
    for dataset_display, model_data in sorted(
        data.items()
    ):
        fig, ax = plt.subplots(figsize=(6, 5))
        is_pheme = (
            "PHEME" in dataset_display.upper()
        )
        for model_display in _MODEL_ORDER:
            if model_display not in model_data:
                continue
            roc_list = model_data[model_display]
            if not roc_list:
                continue
            (
                base_fpr,
                tpr_mean,
                tpr_std,
                auc_mean,
                auc_std,
            ) = _mean_roc(roc_list)
            color = _MODEL_COLORS[model_display]
            ls    = _MODEL_LINESTYLES[
                model_display
            ]
            label = (
                f"{model_display} "
                f"(AUC = {auc_mean:.3f}"
                f" $\\pm$ {auc_std:.3f})"
            )
            ax.plot(
                base_fpr,
                tpr_mean,
                color=color,
                linestyle=ls,
                linewidth=1.8,
                label=label,
            )
            ax.fill_between(
                base_fpr,
                np.clip(
                    tpr_mean - tpr_std, 0, 1
                ),
                np.clip(
                    tpr_mean + tpr_std, 0, 1
                ),
                alpha=0.10,
                color=color,
            )
            summary_rows.append({
                "dataset":  dataset_display,
                "model":    model_display,
                "auc_mean": round(auc_mean, 4),
                "auc_std":  round(auc_std, 4),
                "n_seeds":  len(roc_list),
            })
        # Diagonal de referência
        ax.plot(
            [0, 1], [0, 1],
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label="Aleatório (AUC = 0{,}500)",
        )
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel(
            "Taxa de Falsos Positivos",
            fontsize=11,
        )
        ax.set_ylabel(
            "Taxa de Verdadeiros Positivos",
            fontsize=11,
        )
        ax.set_title(
            f"Curvas ROC - {dataset_display}",
            fontsize=12,
        )
        ax.legend(
            loc="lower right",
            fontsize=8,
            framealpha=0.9,
        )
        ax.grid(
            True,
            linestyle=":",
            alpha=0.4,
        )
        if is_pheme:
            ax.text(
                0.02, 0.05,
                "Nota: PHEME apresenta\n"
                "desbalanceamento severo\n"
                "(~0,3% de exemplos falsos).\n"
                "Interpretar com cautela.",
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="bottom",
                bbox=dict(
                    boxstyle="round",
                    facecolor="wheat",
                    alpha=0.5,
                ),
            )
        plt.tight_layout()
        safe_name = _re.sub(
            r"[^a-zA-Z0-9_]",
            "_",
            dataset_display.lower(),
        )
        for ext in ["pdf", "png"]:
            fig_path = (
                out_path
                / f"roc_curves_exp_a_"
                  f"{safe_name}.{ext}"
            )
            fig.savefig(
                fig_path,
                dpi=300,
                bbox_inches="tight",
            )
            logger.info(
                "STEP 7 | ROC: figura salva: %s",
                fig_path,
            )
        plt.close(fig)
    # Resumo CSV 
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        csv_path = (
            Path("results")
            / "roc_auc_summary_exp_a.csv"
        )
        df_summary.to_csv(
            csv_path, index=False
        )
        logger.info(
            "STEP 7 | ROC: resumo CSV salvo: "
            "%s",
            csv_path,
        )
        # Tabela LaTeX do resumo 
        df_pivot = df_summary.pivot_table(
            index="dataset",
            columns="model",
            values="auc_mean",
            aggfunc="first",
        )
        cols_present = [
            m for m in _MODEL_ORDER
            if m in df_pivot.columns
        ]
        df_pivot = df_pivot[cols_present]
        df_pivot = df_pivot.sort_index()
        n_cols   = len(cols_present)
        col_spec = (
            r">{\raggedright\arraybackslash}"
            r"p{3.0cm}"
            + (
                r">{\centering\arraybackslash}X"
                * n_cols
            )
        )
        tex_lines = [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{AUC-ROC médio por modelo "
            r"e dataset no Experimento~A "
            r"(média sobre 10 sementes). "
            r"O PHEME é apresentado "
            r"separadamente pelo "
            r"desbalanceamento severo da "
            r"classe falsa "
            r"($\approx$0{,}3\% de exemplos "
            r"falsos).}",
            r"\label{tab:roc_auc_exp_a}",
            r"\small",
            r"\begin{tabularx}{\textwidth}{"
            + col_spec + "}",
            r"\hline",
        ]
        header = " & ".join(
            r"\textbf{"
            + _MODEL_ABBREV.get(c, c)
            + "}"
            for c in cols_present
        )
        tex_lines.append(
            r"\textbf{Dataset} & "
            + header
            + r" \\"
        )
        tex_lines.append(r"\hline")
        pheme_rows   = []
        regular_rows = []
        for ds_name, row in df_pivot.iterrows():
            if "PHEME" in str(ds_name).upper():
                pheme_rows.append(
                    (ds_name, row)
                )
            else:
                regular_rows.append(
                    (ds_name, row)
                )
        
        for ds_name, row in regular_rows:
            tex_lines.append(
                _fmt_row(ds_name, row)
            )
        if pheme_rows:
            tex_lines.append(r"\hline")
            for ds_name, row in pheme_rows:
                tex_lines.append(
                    _fmt_row(ds_name, row)
                    + r"$^{\dagger}$"
                )
        tex_lines.extend([
            r"\hline",
            r"\end{tabularx}",
            r"\begin{flushleft}",
            r"\footnotesize",
            r"Nota: BERT refere-se ao "
            r"BERTimbau nos \textit{datasets} "
            r"em português e ao BERT-base nos "
            r"\textit{datasets} em inglês. "
            r"Valores em negrito indicam o "
            r"maior AUC-ROC por linha. "
            r"$^{\dagger}$~PHEME: AUC-ROC "
            r"pouco informativa pelo "
            r"desbalanceamento severo. "
            r"Fonte: dados da pesquisa.",
            r"\end{flushleft}",
            r"\end{table}",
        ])
        tex_path = (
            Path("results")
            / "roc_auc_summary_exp_a.tex"
        )
        tex_path.write_text(
            "\n".join(tex_lines),
            encoding="utf-8",
        )
        logger.info(
            "STEP 7 | ROC: tabela LaTeX "
            "salva: %s",
            tex_path,
        )
    logger.info(
        "STEP 7 | ROC: concluído - "
        "%d dataset(s) processado(s).",
        len(data),
    )

# main()
def main(
    cfg_path: str = "configs/experiment.yaml",
    skip_bert: bool = False,
    skip_bilstm: bool = False,
    skip_lr: bool = False,
    skip_mlp: bool = False,
    only_dataset: Optional[str] = None,
    seeds: Optional[List[int]] = None,
    only_d: bool = False,
    only_validate: bool = False,
    only_aggregate: bool = False,
    only_stats: bool = False,
    only_postprocess: bool = False,
    only_step7: bool = False,
) -> None:
    cfg = load_config(cfg_path)
    active_seeds = seeds if seeds else SEEDS
    # Criar diretórios
    for d in [
        "results/metrics",
        "results/metrics/aggregated",
        "results/models",
        "results/figures",
        "results/training_history",
        "results/predictions",
        "results/logs",
        "results/metrics/statistical_tests",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)
    _run_start = datetime.now()
    # Determinar modo de execução
    only_flags = {
        "--only-validate":    only_validate,
        "--only-aggregate":   only_aggregate,
        "--only-stats":       only_stats,
        "--only-postprocess": only_postprocess,
        "--only-d":           only_d,
        "--only-step7":       only_step7,
    }
    active_only = [
        name for name, val
        in only_flags.items() if val
    ]
    run_mode = (
        active_only[0]
        if active_only
        else "pipeline completo"
    )
    logger.info("=" * 70)
    logger.info(
        "RUN START - %s | modo: %s",
        _run_start.isoformat(), run_mode,
    )
    logger.info(
        "Log: results/logs/run_%s.log",
        _RUN_TIMESTAMP,
    )
    logger.info("=" * 70)
    if len(active_only) > 1:
        logger.error(
            "Conflito de flags: %s são "
            "mutuamente exclusivos. "
            "Use apenas um --only-* por vez.",
            active_only,
        )
        _log_run_end(_run_start)
        return
    # Registrar ambiente
    env = register_environment(
        active_seeds[0],
        output_path=(
            f"results/logs/"
            f"environment_{_RUN_TIMESTAMP}.json"
        ),
    )
    exp_version = get_nested(
        cfg,
        "general", "experiment_version",
        default="N/A",
    )
    logger.info(
        "Python %s | seeds=%s | version=%s",
        env["python_version"],
        active_seeds,
        exp_version,
    )
    # Modelos ativos
    model_names = [
        m for m in MODELS
        if not (
            (m == "bertimbau" and skip_bert)
            or (m == "bilstm"  and skip_bilstm)
            or (
                m == "logistic_regression"
                and skip_lr
            )
            or (m == "mlp" and skip_mlp)
        )
    ]
    logger.info(
        "Active models: %s", model_names
    )
    # Datasets ativos
    active_map = _build_active_datasets(
        only_dataset
    )
    logger.info(
        "Active datasets: %s",
        list(active_map.keys()),
    )

    # Modo --only-validate
    if only_validate:
        logger.info(
            "ONLY STEP 4b - Validating results"
        )
        available_names = set(active_map.keys())
        combined_names  = (
            get_combined_dataset_names(
                available_names
            )
        )
        report = validate_metrics_availability(
            metrics_dir="results/metrics",
            seeds=active_seeds,
            model_names=model_names,
            dataset_names=(
                list(available_names)
                + combined_names
            ),
            exp_prefixes=[
            "EXP_A", "EXP_B", "EXP_C",
            ],
            output_path=(
                "results/metrics/"
                "data_validation_report.json"
            ),
        )
        logger.info(
            "STEP 4b - %d/%d arquivos | "
            "%d ausentes | completo=%s",
            report["present_count"],
            report["expected_count"],
            report["missing_count"],
            report["complete"],
        )
        if not report["complete"]:
            for fname in (
                report["missing_files"][:10]
            ):
                logger.warning(
                    " %s", fname
                )
            if report.get("missing_truncated"):
                logger.warning(
                    "  ... +%d não listados.",
                    report["missing_count"] - 100,
                )
        else:
            logger.info(
                " Todos os arquivos presentes."
            )
        _log_run_end(_run_start)
        return

    # Modo --only-aggregate
    if only_aggregate:
        logger.info(
            "ONLY STEP 5 - Aggregating results"
        )
        metrics_path = Path("results/metrics")
        if not list(
            metrics_path.glob("EXP_*.json")
        ):
            logger.error(
                "STEP 5 - Nenhum EXP_*.json. "
                "Execute o pipeline primeiro."
            )
            _log_run_end(_run_start)
            return
        aggregated_results = (
            aggregate_all_experiments(
                metrics_dir="results/metrics",
                aggregated_dir=(
                    "results/metrics/aggregated"
                ),
                seeds=active_seeds,
                model_names=model_names,
                dataset_names=list(
                    active_map.keys()
                ),
            )
        )
        agg_path  = Path(
            "results/metrics/aggregated"
        )
        agg_files = (
            list(agg_path.glob("*.json"))
            if agg_path.exists()
            else []
        )
        logger.info(
            "STEP 5 - %d arquivo(s) agregados.",
            len(agg_files),
        )
        _log_run_end(_run_start)
        return

    # Modo --only-stats
    if only_stats:
        logger.info(
            "ONLY STEP 6 - Statistical tests"
        )
        agg_path  = Path(
            "results/metrics/aggregated"
        )
        agg_files = (
            list(agg_path.glob("*.json"))
            if agg_path.exists()
            else []
        )
        if not agg_files:
            logger.error(
                "STEP 6 - Nenhum resultado "
                "agregado. Execute "
                "--only-aggregate primeiro."
            )
            _log_run_end(_run_start)
            return
        aggregated_results_stats: Dict[
            str, Any
        ] = {}
        for jf in agg_files:
            try:
                data = json.loads(
                    jf.read_text(
                        encoding="utf-8"
                    )
                )
                aggregated_results_stats[
                    jf.stem
                ] = data
            except Exception as e:
                logger.warning(
                    "STEP 6 | Falha: %s: %s",
                    jf.name, e,
                )
        run_statistical_tests_from_aggregation(
            aggregated_results=(
                aggregated_results_stats
            ),
            seeds=active_seeds,
            cfg=cfg,
            output_dir=(
                "results/metrics/"
                "statistical_tests"
            ),
        )
        _log_run_end(_run_start)
        return

    # Modo --only-postprocess
    if only_postprocess:
        logger.info(
            "ONLY Post-processing: "
            "tables, figures, report"
        )
        _run_postprocess()
        _log_run_end(_run_start)
        return

    # Modo --only-step7
    if only_step7:
        logger.info(
            "ONLY STEP 7 - Text length stats "
            "+ ROC curves"
        )
        # STEP 1 simplificado: carrega apenas
        # os datasets necessários para as
        # estatísticas de comprimento.
        # As curvas ROC leem apenas arquivos
        # de predição já salvos em disco,
        # sem necessidade de carregar os dados.
        raw_step7: Dict[str, pd.DataFrame] = {}
        for ds_name, ds_path in active_map.items():
            try:
                df7 = load_dataset(
                    ds_name, ds_path, cfg=cfg
                )
                # Backup de timestamp
                ts_backup = None
                if "timestamp" in df7.columns:
                    ts_backup = (
                        df7["timestamp"].copy()
                    )
                    df7 = df7.drop(
                        columns=["timestamp"]
                    )
                df7 = preprocess(df7, cfg)
                if ts_backup is not None:
                    df7["timestamp"] = ts_backup
                df7 = remove_exact_duplicates(
                    df7, text_col="text"
                )
                df7 = remove_near_duplicates(
                    df7,
                    text_col="text",
                    similarity_threshold=0.90,
                )
                # near_duplicate_groups() não é calculado aqui:
                # _nd_group é necessário apenas para random_split(), 
                # que não é executado no modo --only-step7.
                raw_step7[ds_name] = df7
                logger.info(
                    "STEP 7 | %s carregado: "
                    "%d exemplos",
                    ds_name, len(df7),
                )
            except Exception as e:
                logger.warning(
                    "STEP 7 | %s: falha ao "
                    "carregar: %s - pulando.",
                    ds_name, e,
                )
        if raw_step7:
            try:
                _compute_text_length_stats(
                    raw=raw_step7,
                    output_dir="results",
                )
            except Exception as e:
                logger.error(
                    "STEP 7a | falhou: %s", e
                )
        else:
            logger.warning(
                "STEP 7 | Nenhum dataset "
                "carregado para estatísticas "
                "de comprimento."
            )
        try:
            _generate_roc_curves(
                seeds=active_seeds,
                model_names=model_names,
                predictions_dir=(
                    "results/predictions"
                ),
                output_dir="results/figures",
            )
        except Exception as e:
            logger.error(
                "STEP 7b | ROC curves "
                "falhou: %s", e
            )
        _log_run_end(_run_start)
        return

    # STEP 1 - Load & preprocess datasets
    logger.info("STEP 1 - Loading datasets")
    raw: Dict[str, pd.DataFrame] = {}
    for ds_name, ds_path in active_map.items():
        try:
            df = load_dataset(ds_name, ds_path, cfg=cfg)
            log_counts(df, ds_name, "após load_dataset")
            assert_valid(df, ds_name)
            log_counts(df, ds_name, "após assert_valid")
            # Backup timestamp
            timestamp_backup = None
            if "timestamp" in df.columns:
                timestamp_backup = (
                    df["timestamp"].copy()
                )
                df = df.drop(
                    columns=["timestamp"]
                )
                logger.info(
                    "%s | timestamp salvo "
                    "(%d válidos)",
                    ds_name,
                    timestamp_backup.notna().sum(),
                )
            df = preprocess(df, cfg)
            log_counts(df, ds_name, "após preprocess")
            # Restaurar timestamp
            if timestamp_backup is not None:
                df["timestamp"] = timestamp_backup
            elif "Data" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    df["Data"], errors="coerce"
                )
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"],
                    errors="coerce",
                )
                logger.info(
                    "%s | timestamp válido: "
                    "%d/%d",
                    ds_name,
                    df["timestamp"].notna().sum(),
                    len(df),
                )
            # file_hash
            try:
                hashes = _hash_dataset_files(
                    ds_name,
                    Path(ds_path),
                )
                if hashes:
                    hash_path = Path(
                        "results/logs/"
                        f"dataset_hashes_"
                        f"{_RUN_TIMESTAMP}.json"
                    )
                    existing = (
                        json.loads(
                            hash_path.read_text(
                                encoding="utf-8"
                            )
                        )
                        if hash_path.exists()
                        else {}
                    )
                    existing[ds_name] = hashes
                    hash_path.write_text(
                        json.dumps(
                            existing, indent=2
                        ),
                        encoding="utf-8",
                    )
            except Exception as e:
                logger.warning(
                    "%s | file_hash falhou: %s",
                    ds_name, e,
                )
            if cfg.get("debug", {}).get(
                "diagnose_duplicates", False
            ):
                df.to_csv(
                    f"results/{ds_name}_before_exact_dedup.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                _diagnose_duplicates(df, ds_name)

            # Deduplicação - exact e near-duplicates removidos ANTES do split
            # para evitar que pares quase-idênticos atravessem treino/teste.
            df = remove_exact_duplicates(df, text_col="text")
            log_counts(df, ds_name, "após duplicatas exatas")
            n_before_nd = len(df)
            df = remove_near_duplicates(
                df, text_col="text",
                similarity_threshold=0.90,
            )
            log_counts(df, ds_name, "após near-duplicates")
            n_removed_nd = n_before_nd - len(df)
            logger.info(
                "%s | near-duplicates removidos antes do "
                "split: %d (%d -> %d). N final: %d.",
                ds_name, n_removed_nd,
                n_before_nd, len(df), len(df),
            )
            # Calcular grupos de near-duplicates no df final para uso no
            # random_split (group_col impede que grupos cruzem partições).
            df = df.reset_index(drop=True)
            df["_nd_group"] = near_duplicate_groups(
                df, text_col="text", similarity_threshold=0.90,
            )
            print_dataset_report(df, ds_name)
            raw[ds_name] = df
            logger.info(
                "Loaded %s: %d rows "
                "(fake=%d, real=%d)",
                ds_name, len(df),
                int((df["label"] == 1).sum()),
                int((df["label"] == 0).sum()),
            )
        except FileNotFoundError as e:
            logger.warning(
                "Skipping '%s': %s",
                ds_name, e,
            )
        except ValueError as e:
            logger.error(
                "Erro ao pré-processar '%s': "
                "%s - dataset ignorado.",
                ds_name, e,
            )
    if len(raw) < 1:
        logger.error(
            "At least 1 dataset required. "
            "Aborting."
        )
        _log_run_end(_run_start)
        return


    # Modo --only-d
    if only_d:
        logger.info(
            "ONLY EXPERIMENT D - "
            "Temporal analysis"
        )
        sklearn_bp_d = _load_or_tune_sklearn(
            raw=raw,
            cfg=cfg,
            tuning_seed=active_seeds[0],
            model_names=[
                m for m in model_names
                if m in {"logistic_regression", "mlp"}
            ],
            metrics_dir="results/metrics",
        )
        mlp_best_params_d = sklearn_bp_d.get("mlp", {})
        lr_best_params_d = sklearn_bp_d.get(
            "logistic_regression", {}
        )
        for seed in active_seeds:
            set_seed(seed)
            for ds_name, df in raw.items():
                if ds_name not in TEMPORAL_DATASETS:
                    continue
                if "timestamp" not in df.columns:
                    logger.warning(
                        "Seed %d | Skipping %s "
                        "(no timestamp column)",
                        seed, ds_name,
                    )
                    continue
                if df["timestamp"].isna().all():
                    logger.warning(
                        "Seed %d | Skipping %s "
                        "(all timestamps NaT)",
                        seed, ds_name,
                    )
                    continue
                logger.info(
                    "Seed %d | %s | "
                    "timestamp válido: %d/%d",
                    seed, ds_name,
                    df["timestamp"].notna().sum(),
                    len(df),
                )
                if seed == active_seeds[0]:
                    logger.info(
                        "Seed %d | EXP_D | %s | "
                        "NOTA: split temporal é "
                        "determinístico - mesmos "
                        "dados para todas as seeds. "
                        "Variação entre seeds "
                        "reflete apenas "
                        "inicialização do modelo.",
                        seed, ds_name,
                    )
                for model_name in model_names:
                    try:
                        run_temporal_experiment(
                            dataset_name=ds_name,
                            df=df,
                            model_name=model_name,
                            cfg=cfg,
                            seed=seed,
                            model_kwargs=_build_model_cfg(
                                cfg, model_name, ds_name,
                                mlp_best_params_d,
                                lr_best_params_d,
                            ),
                        )
                    except ValueError as e:
                        logger.warning(
                            "EXP_D | %s | %s | "
                            "seed=%d: %s - pulando.",
                            ds_name, model_name,
                            seed, e,
                        )
                    except Exception as e:
                        logger.error(
                            "EXP_D | %s | %s | "
                            "seed=%d: %s",
                            ds_name, model_name,
                            seed, e,
                        )
        logger.info(
            "ONLY EXPERIMENT D COMPLETED"
        )
        _log_run_end(_run_start)
        return

    # STEP 2 - Cross-dataset leakage detection
    logger.info(
        "STEP 2 - Cross-dataset leakage "
        "detection"
    )
    leakage = detect_cross_dataset_leakage(raw)
    with open(
        "results/metrics/leakage_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                f"{a}___{b}": v
                for (a, b), v in leakage.items()
            },
            f,
            indent=2,
            default=str,
        )

    # STEP 3A - Sklearn hyperparameter tuning
    sklearn_models_in_run = [
        m for m in model_names
        if m in {
            "logistic_regression", "mlp"
        }
    ]
    sklearn_best_params: Dict[
        str, Dict[str, Dict[str, Any]]
    ] = {m: {} for m in sklearn_models_in_run}
    if sklearn_models_in_run:
        logger.info(
            "STEP 3A - Sklearn GridSearchCV: "
            "ONE tuning run per dataset | "
            "modelos=%s",
            sklearn_models_in_run,
        )
        sklearn_best_params = (
            _load_or_tune_sklearn(
                raw=raw,
                cfg=cfg,
                tuning_seed=active_seeds[0],
                model_names=sklearn_models_in_run,
                metrics_dir="results/metrics",
            )
        )
        logger.info(
            "Sklearn best_params prontos: %s",
            {
                m: list(v.keys())
                for m, v in
                sklearn_best_params.items()
            },
        )
    else:
        logger.info(
            "STEP 3A - Nenhum modelo sklearn "
            "ativo - tuning pulado."
        )
    mlp_best_params: Dict[
        str, Dict[str, Any]
    ] = sklearn_best_params.get("mlp", {})
    lr_best_params: Dict[
        str, Dict[str, Any]
    ] = sklearn_best_params.get(
        "logistic_regression", {}
    )
    logger.info(
        "STEP 3A - mlp_best_params: %d "
        "dataset(s) | "
        "lr_best_params: %d dataset(s)",
        len(mlp_best_params),
        len(lr_best_params),
    )

    # STEP 3B - Multi-seed execution
    logger.info(
        "STEP 3B - Multi-seed execution: "
        "seeds=%s",
        active_seeds,
    )
    all_intra_f1: Dict[
        int, Dict[Tuple[str, str], float]
    ] = {}
    for seed in active_seeds:
        seed_intra = _run_single_seed(
            seed=seed,
            raw=raw,
            model_names=model_names,
            cfg=cfg,
            mlp_best_params=mlp_best_params,
            lr_best_params=lr_best_params,
            active_seeds=active_seeds,
        )
        all_intra_f1[seed] = seed_intra

    # STEP 4 - Per-seed F1 summary
    logger.info("STEP 4 - Per-seed F1 summary")
    for model_name in model_names:
        for ds_name in raw:
            values = [
                all_intra_f1[s].get(
                    (model_name, ds_name),
                    float("nan"),
                )
                for s in active_seeds
            ]
            valid = [
                v for v in values
                if not np.isnan(v)
            ]
            if valid:
                mean_f1 = float(np.mean(valid))
                std_f1  = (
                    float(
                        np.std(valid, ddof=1)
                    )
                    if len(valid) > 1
                    else 0.0
                )
                seed_strs = "  ".join(
                    f"seed{s}={v:.4f}"
                    for s, v in zip(
                        active_seeds, values
                    )
                )
                logger.info(
                    "%-20s %-20s | %s | "
                    "F1=%.4f±%.4f",
                    model_name, ds_name,
                    seed_strs,
                    mean_f1, std_f1,
                )

    # STEP 4b - Validate results
    logger.info(
        "STEP 4b - Validating results "
        "availability"
    )
    combined_names = get_combined_dataset_names(
        set(raw.keys())
    )
    # Datasets válidos para EXP_D
    TEMPORAL_VALID = {
        ds for ds in raw.keys()
        if (
            "timestamp" in raw[ds].columns
            and (
                raw[ds]["timestamp"]
                .notna().sum() >= 50
            )
            and ds in TEMPORAL_DATASETS
            and raw[ds].loc[
                raw[ds]["timestamp"].notna(),
                "label",
            ].nunique() >= 2
        )
    }
    logger.info(
        "STEP 4b - datasets válidos para "
        "EXP_D: %s",
        sorted(TEMPORAL_VALID),
    )
    # Validar EXP_A, EXP_B, EXP_C
    report = validate_metrics_availability(
        metrics_dir="results/metrics",
        seeds=active_seeds,
        model_names=model_names,
        dataset_names=(
            list(raw.keys()) + combined_names
        ),
        exp_prefixes=[
            "EXP_A", "EXP_B", "EXP_C",
        ],
        output_path=(
            "results/metrics/"
            "data_validation_report.json"
        ),
    )
    logger.info(
        "STEP 4b - %d/%d arquivos (%s)",
        report["present_count"],
        report["expected_count"],
        " COMPLETO"
        if report["complete"]
        else (
            f" {report['missing_count']} "
            f"AUSENTES"
        ),
    )
    if not report["complete"]:
        for fname in (
            report["missing_files"][:10]
        ):
            logger.warning(
                " %s", fname
            )
        if report.get("missing_truncated"):
            logger.warning(
                "  ... +%d não listados.",
                report["missing_count"] - 100,
            )
    # Validar EXP_D separadamente
    if TEMPORAL_VALID:
        report_d = validate_metrics_availability(
            metrics_dir="results/metrics",
            seeds=active_seeds,
            model_names=model_names,
            dataset_names=list(TEMPORAL_VALID),
            exp_prefixes=["EXP_D"],
            output_path=(
                "results/metrics/"
                "data_validation_report_expD.json"
            ),
        )
        logger.info(
            "STEP 4b EXP_D - %d/%d arquivos "
            "(%s) | datasets: %s",
            report_d["present_count"],
            report_d["expected_count"],
            "OK" if report_d["complete"] else "WARNING",
            sorted(TEMPORAL_VALID),
        )
        if not report_d["complete"]:
            for fname in (
                report_d["missing_files"][:10]
            ):
                logger.warning(
                    " EXP_D: %s", fname
                )
    else:
        logger.warning(
            "STEP 4b - Nenhum dataset válido "
            "para EXP_D (timestamps insuficientes "
            "ou distribuição de classes inválida)."
        )

    # STEP 5 - Aggregate multi-seed results
    logger.info(
        "STEP 5 - Aggregating results "
        "across seeds"
    )
    for exp in ["A", "B", "C", "D"]:
        files = list(
            Path("results/metrics").glob(
                f"EXP_{exp}_*.json"
            )
        )
        logger.info(
            "STEP 5 | EXP_%s: %d arquivo(s)",
            exp, len(files),
        )
    for seed in active_seeds:
        seed_files = list(
            Path("results/metrics").glob(
                f"*_seed{seed}.json"
            )
        )
        if not seed_files:
            logger.warning(
                "STEP 5 | seed=%d: NENHUM "
                "arquivo - resultados "
                "incompletos.",
                seed,
            )
    aggregated_results = (
        aggregate_all_experiments(
            metrics_dir="results/metrics",
            aggregated_dir=(
                "results/metrics/aggregated"
            ),
            seeds=active_seeds,
            model_names=model_names,
            dataset_names=list(raw.keys()),
        )
    )
    agg_path  = Path(
        "results/metrics/aggregated"
    )
    agg_files = (
        list(agg_path.glob("*.json"))
        if agg_path.exists()
        else []
    )
    logger.info(
        "STEP 5 - Agregação concluída: "
        "%d arquivo(s) em %s",
        len(agg_files), agg_path,
    )
    if not agg_files:
        logger.warning(
            "STEP 5 - Nenhum arquivo agregado. "
            "Verificar "
            "aggregate_all_experiments()."
        )

    # STEP 6 - Statistical tests
    logger.info("STEP 6 - Statistical tests")
    if not agg_files:
        logger.error(
            "STEP 6 - Nenhum resultado "
            "agregado. Pulando testes."
        )
    else:
        statistical_results = (
            run_statistical_tests_from_aggregation(
                aggregated_results=(
                    aggregated_results
                ),
                seeds=active_seeds,
                cfg=cfg,
                output_dir=(
                    "results/metrics/"
                    "statistical_tests"
                ),
            )
        )
        stats_path = Path(
            "results/metrics/statistical_tests"
        )
        stats_files = (
            list(stats_path.glob("*.json"))
            if stats_path.exists()
            else []
        )
        logger.info(
            "STEP 6 - Concluído: "
            "%d resultado(s). Chaves: %s",
            len(statistical_results),
            list(statistical_results.keys()),
        )
        if not stats_files:
            logger.warning(
                "STEP 6 - Nenhum arquivo "
                "salvo em %s.",
                stats_path,
            )


    # Post-processing
    _run_postprocess()

    # STEP 7 - Text length stats + ROC curves
    logger.info(
        "STEP 7 - Text length statistics "
        "and ROC curves"
    )
    # 7a - Estatísticas de comprimento
    # Usa os DataFrames já carregados no STEP 1
    # Não recarrega nem reprocessa os datasets
    try:
        _compute_text_length_stats(
            raw=raw,
            output_dir="results",
        )
    except Exception as e:
        logger.error(
            "STEP 7a | text length stats "
            "falhou: %s",
            e,
        )
    # 7b - Curvas ROC do EXP_A
    # Lê os arquivos de predição já salvos
    # Não reexecuta nenhum experimento
    try:
        _generate_roc_curves(
            seeds=active_seeds,
            model_names=model_names,
            predictions_dir=(
                "results/predictions"
            ),
            output_dir="results/figures",
        )
    except Exception as e:
        logger.error(
            "STEP 7b | ROC curves "
            "falhou: %s",
            e,
        )
    _log_run_end(_run_start)
    logger.info(
        "ALL EXPERIMENTS COMPLETE. "
        "Results in results/"
    )

# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Pipeline experimental completo - "
            "6 datasets, 4 modelos, 10 seeds"
        )
    )
    ap.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help=(
            "Caminho para o arquivo de "
            "configuração YAML"
        ),
    )
    ap.add_argument(
        "--skip-bert",
        action="store_true",
        help="Pula BERTimbau",
    )
    ap.add_argument(
        "--skip-bilstm",
        action="store_true",
        help="Pula BiLSTM",
    )
    ap.add_argument(
        "--skip-lr",
        action="store_true",
        help="Pula Logistic Regression",
    )
    ap.add_argument(
        "--skip-mlp",
        action="store_true",
        help="Pula MLP",
    )
    ap.add_argument(
        "--only-validate",
        action="store_true",
        help="Executa somente STEP 4b.",
    )
    ap.add_argument(
        "--only-d",
        action="store_true",
        help="Executa somente EXP_D.",
    )
    ap.add_argument(
        "--only-aggregate",
        action="store_true",
        help="Executa somente STEP 5.",
    )
    ap.add_argument(
        "--only-stats",
        action="store_true",
        help="Executa somente STEP 6.",
    )

    ap.add_argument(
        "--only-postprocess",
        action="store_true",
        help=(
            "Executa somente post-processing."
        ),
    )

    ap.add_argument(
        "--only-step7",
        action="store_true",
        help=(
            "Executa somente STEP 7 "
            "(text length stats + ROC curves). "
            "Requer EXP_A já executado."
        ),
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help=(
            "Rodar apenas 1 dataset: "
            "faketruebr|fakebr|fakerecogna|"
            "fakenewsnet|"
            "constraint_covid19|pheme"
        ),
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Lista de seeds "
            "(padrão: 42 43 44 45 46). "
            "Ex: --seeds 42 43"
        ),
    )
    args = ap.parse_args()
    main(
        cfg_path=args.config,
        skip_bert=args.skip_bert,
        skip_bilstm=args.skip_bilstm,
        skip_lr=args.skip_lr,
        skip_mlp=args.skip_mlp,
        only_dataset=args.dataset,
        seeds=args.seeds,
        only_d=args.only_d,
        only_validate=args.only_validate,
        only_aggregate=args.only_aggregate,
        only_stats=args.only_stats,
        only_postprocess=args.only_postprocess,
        only_step7=args.only_step7,        
    )