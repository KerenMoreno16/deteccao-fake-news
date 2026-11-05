"""
evaluate.py - Aggregate results from saved JSON files
into summary tables.
Correções aplicadas:
  - load_all_results(): verificação de existência do
    diretório; log de arquivos encontrados/ignorados.
  - build_intra_dataset_table(): alerta sobre células
    com número diferente de seeds.
  - build_cross_dataset_matrix(): regex não alterado
    (já corrigido na versão anterior).
  - compute_generalization_gap(): log de gaps NaN.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from src.utils.logging import get_logger
logger = get_logger(__name__)
def load_all_results(
    results_dir: str = "results/metrics",
) -> pd.DataFrame:
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning(
            "Diretório de resultados não encontrado: "
            "'%s'. Retornando DataFrame vazio.",
            results_dir,
        )
        return pd.DataFrame()
    if not results_path.is_dir():
        raise ValueError(
            f"'{results_dir}' existe mas não é "
            f"um diretório."
        )
    rows          : List[Dict] = []
    files_found   : int        = 0
    files_skipped : int        = 0
    # Contagem de arquivos em subdiretórios -
    # executada UMA VEZ antes do loop principal.
    # Apenas informativa: o loop abaixo usa glob()
    # (não rglob) e não entra em subdiretórios.
    total_in_subdirs = sum(
        1 for p in results_path.rglob("EXP_*.json")
        if p.parent != results_path
        and "_predictions" not in p.stem
    )
    if total_in_subdirs > 0:
        logger.debug(
            "load_all_results: %d arquivo(s) em "
            "subdiretórios ignorados (não varridos).",
            total_in_subdirs,
        )
    # Loop principal - apenas arquivos diretos
    for p in sorted(
        results_path.glob("EXP_*.json")
    ):
        if "_predictions" in p.stem:
            logger.debug(
                "Pulando predições: %s", p.name
            )
            files_skipped += 1
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(
                "JSON inválido ignorado: %s - %s",
                p.name, e,
            )
            files_skipped += 1
            continue
        except OSError as e:
            logger.warning(
                "Erro ao ler: %s - %s",
                p.name, e,
            )
            files_skipped += 1
            continue
        row = {
            "experiment_id": data.get(
                "experiment_id"
            ),
            "model":      data.get("model"),
            "dataset":    data.get("dataset"),
            "split_type": data.get(
                "split_type", "random"
            ),
            "train_size": data.get("train_size"),
            "test_size":  data.get("test_size"),
        }
        m = data.get("metrics", {})
        for key in [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
            "roc_auc",
            "pr_auc",
            "mcc",
            "f1_macro_ci_lo",
            "f1_macro_ci_hi",
            "train_time_seconds",
            "inference_time_seconds",
        ]:
            row[key] = m.get(key)
        rows.append(row)
        files_found += 1
    df = pd.DataFrame(rows)
    logger.info(
        "load_all_results: %d carregado(s), "
        "%d ignorado(s) em '%s'",
        files_found, files_skipped, results_dir,
    )
    if df.empty:
        logger.warning(
            "Nenhum resultado encontrado em '%s'. "
            "Verifique se os experimentos foram "
            "executados.",
            results_dir,
        )
    return df
def build_intra_dataset_table(
    results: pd.DataFrame,
    expected_seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Pivot: rows = models, columns = datasets,
    values = F1-macro (média sobre seeds).
    Parâmetros
    ----------
    results        : DataFrame de load_all_results().
    expected_seeds : lista de seeds esperadas.
                     Se None, usa o máximo encontrado.
    Retorna
    -------
    DataFrame pivotado com médias de F1-macro.
    DataFrame vazio se não houver dados.
    Notas
    -----
    aggfunc="mean" calcula a média sobre todas as seeds
    presentes por par (model, dataset).
    Células com menos seeds que o esperado são alertadas
    via log - a média ainda é calculada mas é menos
    estável.
    """
    filtered = results[
        results["split_type"] == "random"
    ].copy()
    if filtered.empty:
        logger.warning(
            "build_intra_dataset_table: nenhum "
            "resultado com split_type='random'."
        )
        return pd.DataFrame()
    # CORRIGIDO: verifica contagem de seeds por célula
    seed_counts = (
        filtered
        .groupby(["model", "dataset"])["experiment_id"]
        .nunique()
        .reset_index()
        .rename(columns={"experiment_id": "n_seeds"})
    )
    if expected_seeds is not None:
        n_expected = len(expected_seeds)
    else:
        n_expected = int(seed_counts["n_seeds"].max())
    incomplete = seed_counts[
        seed_counts["n_seeds"] < n_expected
    ]
    if not incomplete.empty:
        for _, row in incomplete.iterrows():
            logger.warning(
                "build_intra_dataset_table: "
                "model='%s', dataset='%s' tem "
                "%d/%d seeds. "
                "Média menos estável.",
                row["model"],
                row["dataset"],
                int(row["n_seeds"]),
                n_expected,
            )
    return filtered.pivot_table(
        index="model",
        columns="dataset",
        values="f1_macro",
        aggfunc="mean",
    ).round(4)
def build_cross_dataset_matrix(
    results: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Constrói a matriz treino->teste de F1-macro
    para um único modelo.
    Parâmetros
    ----------
    results    : DataFrame de load_all_results().
    model_name : nome do modelo a filtrar.
    Retorna
    -------
    DataFrame pivotado (train_ds × test_ds).
    DataFrame vazio se não houver dados.
    Notas
    -----
    O regex captura train_ds e test_ds do experiment_id
    no formato:
        EXP_B_{model}_TRAIN_{train}_TEST_{test}_seed{N}
    O uso de (.+) greedy pode capturar incorretamente
    se train_ds ou test_ds contiverem "_TEST_" ou
    "_TRAIN_" como substring. Os seis datasets atuais
    não têm esse problema.
    """
    cross = results[
        (results["model"] == model_name)
        & (results["split_type"] == "cross")
    ].copy()
    if cross.empty:
        logger.warning(
            "build_cross_dataset_matrix: "
            "nenhum resultado cross para '%s'.",
            model_name,
        )
        return pd.DataFrame()
    extracted = cross["experiment_id"].str.extract(
        r"EXP_B_[^_]+_TRAIN_(.+)_TEST_(.+)_seed\d+"
    )
    extracted.columns = ["train_ds", "test_ds"]
    cross[["train_ds", "test_ds"]] = extracted
    failed_mask = (
        cross[["train_ds", "test_ds"]]
        .isna()
        .any(axis=1)
    )
    if failed_mask.any():
        failed_ids = cross.loc[
            failed_mask, "experiment_id"
        ].tolist()
        logger.warning(
            "build_cross_dataset_matrix: "
            "regex não capturou %d experiment_id(s): "
            "%s",
            len(failed_ids),
            failed_ids[:5],
        )
        cross = cross.dropna(
            subset=["train_ds", "test_ds"]
        )
    if cross.empty:
        return pd.DataFrame()
    return cross.pivot_table(
        index="train_ds",
        columns="test_ds",
        values="f1_macro",
        aggfunc="mean",
    ).round(4)
def compute_generalization_gap(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Para cada par (model, dataset) calcula:
        gap = F1_intra - média(F1_cross quando
              esse dataset é usado como TEST)
    Parâmetros
    ----------
    results : DataFrame de load_all_results().
    Retorna
    -------
    DataFrame com colunas:
        model, dataset, f1_intra, f1_cross,
        generalization_gap
    Ordenado por generalization_gap decrescente.
    Notas metodológicas
    -------------------
    A agregação cross é feita em dois passos:
      1. Média por seed (para cada model × test_ds)
      2. Média geral entre seeds
    Isso é equivalente à média simples quando todas
    as seeds têm o mesmo número de observações.
    Com seeds faltantes, a média pode ser distorcida -
    verificar log de avisos de seeds incompletas.
    O gap positivo indica que o modelo performa melhor
    no próprio dataset de treino do que em datasets
    externos (esperado).
    Gap negativo pode indicar dataset de treino muito
    pequeno ou específico.
    """
    intra = (
        results[results["split_type"] == "random"]
        .groupby(["model", "dataset"])["f1_macro"]
        .mean()
        .reset_index()
        .rename(columns={"f1_macro": "f1_intra"})
    )
    cross = results[
        results["split_type"] == "cross"
    ].copy()
    if cross.empty:
        logger.warning(
            "compute_generalization_gap: "
            "nenhum resultado cross-dataset. "
            "Gap não calculável."
        )
        return intra.assign(
            f1_cross=None,
            generalization_gap=None,
        )
    # Extrai train_ds, test_ds e seed do experiment_id
    extracted = cross["experiment_id"].str.extract(
        r"EXP_B_[^_]+_TRAIN_(.+)_TEST_(.+)_seed(\d+)"
    )
    extracted.columns = [
        "train_ds", "test_ds", "seed"
    ]
    cross[["train_ds", "test_ds", "seed"]] = extracted
    cross["seed"] = pd.to_numeric(
        cross["seed"], errors="coerce"
    )
    failed_mask = (
        cross[["train_ds", "test_ds"]]
        .isna()
        .any(axis=1)
    )
    if failed_mask.any():
        logger.warning(
            "compute_generalization_gap: "
            "regex não capturou %d registro(s). "
            "Verifique o formato do experiment_id.",
            failed_mask.sum(),
        )
        cross = cross.dropna(
            subset=["train_ds", "test_ds"]
        )
    if cross.empty:
        return intra.assign(
            f1_cross=None,
            generalization_gap=None,
        )
    # Agregação em dois passos:
    # 1. média por seed (model × test_ds)
    # 2. média entre seeds
    cross_agg = (
        cross
        .groupby(
            ["model", "test_ds", "seed"]
        )["f1_macro"]
        .mean()
        .groupby(["model", "test_ds"])
        .mean()
        .reset_index()
        .rename(
            columns={
                "test_ds":  "dataset",
                "f1_macro": "f1_cross",
            }
        )
    )
    merged = intra.merge(
        cross_agg,
        on=["model", "dataset"],
        how="left",
    )
    merged["generalization_gap"] = (
        merged["f1_intra"] - merged["f1_cross"]
    ).round(4)
    # CORRIGIDO: loga pares com gap NaN
    nan_gaps = merged[
        merged["generalization_gap"].isna()
    ]
    if not nan_gaps.empty:
        for _, row in nan_gaps.iterrows():
            logger.warning(
                "compute_generalization_gap: "
                "gap NaN para model='%s', "
                "dataset='%s' - sem dados cross "
                "onde este dataset é TEST. "
                "Verifique EXP_B.",
                row["model"],
                row["dataset"],
            )
    return merged.sort_values(
        "generalization_gap",
        ascending=False,
    ).reset_index(drop=True)