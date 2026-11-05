"""
src/utils/aggregation.py — Central multi-seed results
aggregation.
Collects per-seed JSON metric files, computes
mean ± std (ddof=1) for every scalar metric, and
produces:
  - Per-experiment aggregated JSON
    (results/metrics/aggregated/<exp_id>.json)
  - Per-experiment-type summary CSV
    (results/metrics/aggregated/<type>_summary.csv)
Separação metodológica:
  - Bootstrap CI  -> variabilidade dentro de uma
                    execução (resampling do test set)
  - Seed std      -> variabilidade entre 5 execuções
                    independentes
Ambos são preservados na saída.
"""
from __future__ import annotations
import json
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils.metrics import (
    aggregate_seed_results,
    save_aggregated_results,
    _SCALAR_METRICS,
)
from src.utils.logging import get_logger
logger = get_logger(__name__)

def _load_json(path: Path) -> Dict[str, Any]:
    """
    Carrega arquivo JSON com tratamento de erros.
    Retorna dict vazio e loga aviso se o arquivo
    não existir ou estiver malformado.
    """
    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        warnings.warn(f"File not found: {path}")
        return {}
    except json.JSONDecodeError as exc:
        warnings.warn(
            f"Malformed JSON — {path}: {exc}"
        )
        return {}
    
def _seed_files(
    metrics_dir: Path,
    base_id: str,
    seeds: List[int],
) -> List[Optional[Path]]:
    """
    Retorna lista (mesma ordem que seeds) de Paths
    para arquivos no formato
    ``<base_id>_seed{seed}.json``.
    Arquivos inexistentes são representados como None.
    """
    files: List[Optional[Path]] = []
    for seed in seeds:
        candidate = (
            metrics_dir / f"{base_id}_seed{seed}.json"
        )
        files.append(
            candidate if candidate.exists() else None
        )
    return files

def _aggregate_exp_type(
    metrics_dir: Path,
    aggregated_dir: Path,
    seeds: List[int],
    exp_prefix: str,
    id_extractor: Optional[
        Callable[[Dict[str, Any]], Dict[str, Any]]
    ],
    summary_name: str,
) -> pd.DataFrame:
    """
    Agregador genérico para um prefixo de experimento.
    Varre metrics_dir por arquivos correspondentes a
    ``{exp_prefix}_*_seed{seeds[0]}.json``, determina
    os base IDs e agrega todas as seeds.
    Reporta explicitamente quais seeds estão ausentes
    antes de pular um experimento.
    Parâmetros
    ----------
    metrics_dir    : diretório com JSONs por seed.
    aggregated_dir : diretório de saída.
    seeds          : lista de seeds usadas.
    exp_prefix     : prefixo do experimento (ex: "EXP_A").
    id_extractor   : função opcional para extrair
                     campos adicionais do JSON âncora.
    summary_name   : nome do CSV de resumo.
    Retorna
    -------
    DataFrame com uma linha por combinação
    (model, dataset).
    """
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    first_seed   = seeds[0]
    pattern      = f"{exp_prefix}_*_seed{first_seed}.json"
    anchor_files = sorted(metrics_dir.glob(pattern))
    if not anchor_files:
        logger.debug(
            "Nenhum arquivo para padrão '%s' em %s",
            pattern, metrics_dir,
        )
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for anchor in anchor_files:
        # Deriva base_id removendo _seed{N}.json
        base_id = re.sub(
            r"_seed\d+$", "", anchor.stem
        )
        seed_path_list = _seed_files(
            metrics_dir, base_id, seeds
        )
        # Reportar seeds ausentes explicitamente
        missing_seeds = [
            s for p, s in zip(seed_path_list, seeds)
            if p is None
        ]
        if missing_seeds:
            logger.warning(
                "Seeds ausentes para '%s': %s — "
                "agregação usará apenas seeds "
                "disponíveis.",
                base_id, missing_seeds,
            )
        file_paths = [
            str(p) if p is not None
            else str(
                metrics_dir
                / f"{base_id}_seed{s}.json"
            )
            for p, s in zip(seed_path_list, seeds)
        ]
        agg = aggregate_seed_results(
            file_paths, seeds
        )
        anchor_data = _load_json(anchor)
        if not anchor_data:
            logger.warning(
                "Arquivo âncora vazio ou corrompido: "
                "%s — model/dataset podem ficar "
                "em branco.",
                anchor.name,
            )
        model   = anchor_data.get("model", "")
        dataset = anchor_data.get(
            "dataset",
            anchor_data.get("train_dataset", ""),
        )
        if not model:
            logger.warning(
                "Campo 'model' ausente em %s",
                anchor.name,
            )
        if not dataset:
            logger.warning(
                "Campo 'dataset' ausente em %s",
                anchor.name,
            )
        # Salvar JSON agregado individual
        agg_json_path = (
            aggregated_dir / f"{base_id}_aggregated.json"
        )
        save_aggregated_results(
            aggregated=agg,
            path=str(agg_json_path),
            experiment_prefix=base_id,
            model=model,
            dataset=dataset,
        )
        logger.info(
            "Agregado %d seeds -> %s",
            len(seeds), agg_json_path.name,
        )
        # Flatten para CSV de resumo
        row: Dict[str, Any] = {
            "experiment_prefix": base_id,
            "model":             model,
            "dataset":           dataset,
        }
        extra = (
            id_extractor(anchor_data)
            if id_extractor is not None
            else {}
        )
        row.update(extra)
        for metric, stats in agg["metrics"].items():
            if stats["mean"] is not None:
                row[f"{metric}_mean"] = round(
                    stats["mean"], 6
                )
                row[f"{metric}_std"] = round(
                    stats["std"], 6
                )
                row[f"{metric}_formatted"] = (
                    agg["metrics_formatted"][metric]
                )
                for s, v in zip(
                    seeds, stats["values"]
                ):
                    row[f"{metric}_seed{s}"] = v
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df       = pd.DataFrame(rows)
    csv_path = aggregated_dir / f"{summary_name}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(
        "Summary CSV salvo: %s (%d linhas)",
        csv_path.name, len(df),
    )
    return df

def aggregate_experiment_a(
    metrics_dir: str,
    aggregated_dir: str,
    seeds: List[int],
) -> pd.DataFrame:
    """Agrega resultados do Experimento A (intra-dataset)."""
    return _aggregate_exp_type(
        metrics_dir=Path(metrics_dir),
        aggregated_dir=Path(aggregated_dir),
        seeds=seeds,
        exp_prefix="EXP_A",
        id_extractor=None,
        summary_name="experiment_A_summary",
    )
def aggregate_experiment_b(
    metrics_dir: str,
    aggregated_dir: str,
    seeds: List[int],
) -> pd.DataFrame:
    """Agrega resultados do Experimento B (cross-dataset)."""
    def _extract(
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "train_dataset": data.get(
                "train_dataset", ""
            ),
            "test_dataset":  data.get(
                "test_dataset", ""
            ),
        }
    return _aggregate_exp_type(
        metrics_dir=Path(metrics_dir),
        aggregated_dir=Path(aggregated_dir),
        seeds=seeds,
        exp_prefix="EXP_B",
        id_extractor=_extract,
        summary_name="experiment_B_summary",
    )
def aggregate_experiment_c(
    metrics_dir: str,
    aggregated_dir: str,
    seeds: List[int],
) -> pd.DataFrame:
    """Agrega resultados do Experimento C (combined PT/EN)."""
    return _aggregate_exp_type(
        metrics_dir=Path(metrics_dir),
        aggregated_dir=Path(aggregated_dir),
        seeds=seeds,
        exp_prefix="EXP_C",
        id_extractor=None,
        summary_name="experiment_C_summary",
    )
def aggregate_experiment_d(
    metrics_dir: str,
    aggregated_dir: str,
    seeds: List[int],
) -> pd.DataFrame:
    """Agrega resultados do Experimento D (temporal)."""
    return _aggregate_exp_type(
        metrics_dir=Path(metrics_dir),
        aggregated_dir=Path(aggregated_dir),
        seeds=seeds,
        exp_prefix="EXP_D",
        id_extractor=None,
        summary_name="experiment_D_summary",
    )

def aggregate_all_experiments(
    metrics_dir: str,
    aggregated_dir: str,
    seeds: List[int],
    model_names: Optional[List[str]] = None,
    dataset_names: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Executa agregação para todos os experimentos (A–F).
    Parâmetros
    ----------
    metrics_dir    : diretório com JSONs por seed.
    aggregated_dir : diretório de saída.
    seeds          : lista de seeds usadas.
    model_names    : filtro informacional (não usado).
    dataset_names  : filtro informacional (não usado).
    Retorna
    -------
    dict {"A": DataFrame, "B": DataFrame, ...}
    """
    logger.info(
        "Agregando experimentos — seeds=%s | "
        "metrics_dir=%s",
        seeds, metrics_dir,
    )
    results: Dict[str, pd.DataFrame] = {}
    results["A"] = aggregate_experiment_a(
        metrics_dir, aggregated_dir, seeds
    )
    results["B"] = aggregate_experiment_b(
        metrics_dir, aggregated_dir, seeds
    )
    results["C"] = aggregate_experiment_c(
        metrics_dir, aggregated_dir, seeds
    )
    results["D"] = aggregate_experiment_d(
        metrics_dir, aggregated_dir, seeds
    )

    if not results["A"].empty:
        monograph_cols = [
            "model",
            "dataset",
            "accuracy_formatted",
            "precision_macro_formatted",
            "recall_macro_formatted",
            "f1_macro_formatted",
            "mcc_formatted",
            "roc_auc_formatted",
        ]
        available = [
            c for c in monograph_cols
            if c in results["A"].columns
        ]
        mono_df = results["A"][available].copy()
        rename_map = {
            "model":                     "Model",
            "dataset":                   "Dataset",
            "accuracy_formatted":        "Accuracy",
            "precision_macro_formatted": "Precision",
            "recall_macro_formatted":    "Recall",
            "f1_macro_formatted":        "F1-macro",
            "mcc_formatted":             "MCC",
            "roc_auc_formatted":         "ROC-AUC",
        }
        mono_df   = mono_df.rename(columns=rename_map)
        mono_path = (
            Path(aggregated_dir)
            / "monograph_table_A.csv"
        )
        mono_df.to_csv(mono_path, index=False)
        logger.info(
            "Tabela monografia salva: %s",
            mono_path.name,
        )
    _log_summary(results["A"], seeds)
    return results

def _log_summary(
    df: pd.DataFrame,
    seeds: List[int],
) -> None:
    """
    Loga resumo condensado por modelo/dataset
    do F1-macro.
    """
    if df.empty:
        return
    logger.info("=" * 70)
    logger.info(
        "MULTI-SEED AGGREGATION SUMMARY "
        "(Experiment A — Intra-dataset)"
    )
    logger.info(
        "%-22s %-15s  %-25s  %s",
        "Model", "Dataset",
        "F1-macro (mean±std)", "Seeds",
    )
    logger.info("-" * 70)
    seed_cols = [
        f"f1_macro_seed{s}" for s in seeds
    ]
    for row in df.itertuples(index=False):
        seed_vals = []
        for s, col in zip(seeds, seed_cols):
            val = getattr(row, col, None)
            if val is not None and not (
                isinstance(val, float)
                and np.isnan(val)
            ):
                seed_vals.append(
                    f"s{s}={val:.4f}"
                )
        f1_fmt = getattr(
            row, "f1_macro_formatted", "N/A"
        )
        model  = str(
            getattr(row, "model", "")
        )[:22]
        ds     = str(
            getattr(row, "dataset", "")
        )[:15]
        logger.info(
            "%-22s %-15s  %-25s  %s",
            model, ds, f1_fmt,
            "  ".join(seed_vals),
        )
    logger.info("=" * 70)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Agregar resultados multi-seed"
    )
    ap.add_argument(
        "--metrics-dir",
        default="results/metrics",
    )
    ap.add_argument(
        "--aggregated-dir",
        default="results/metrics/aggregated",
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    )
    args = ap.parse_args()
    aggregate_all_experiments(
        metrics_dir=args.metrics_dir,
        aggregated_dir=args.aggregated_dir,
        seeds=args.seeds,
    )