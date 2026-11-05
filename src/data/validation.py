"""
validation.py — Schema validation and basic quality
checks for loaded datasets.
Correções aplicadas:
  - DATASET_LANGUAGE adicionado para filtro de idioma
    no EXP_B.
  - validate_minimum_samples(): contagem por classe
    usa operação vetorizada em vez de .apply().
  - validate_metrics_availability(): EXP_B ignora
    pares cross-idioma; fallback TEMPORAL_DATASETS
    documentado como limitação conhecida.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from src.utils.logging import get_logger
logger = get_logger(__name__)
# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["text", "label"]
VALID_LABELS     = {0, 1}
DATASET_MIN_SAMPLES: Dict[str, int] = {
    "fakenewsnet":        100,
    "faketruebr":          50,
    "fakebr":              50,
    "fakerecogna":         50,
    "pheme":              500,
    "constraint_covid19": 500,
}
DEFAULT_MIN_SAMPLES = 20
# Mapeamento de idioma por dataset
# Usado por validate_metrics_availability() para
# filtrar pares cross-idioma no EXP_B
DATASET_LANGUAGE: Dict[str, str] = {
    "faketruebr":         "pt",
    "fakebr":             "pt",
    "fakerecogna":        "pt",
    "fakenewsnet":        "en",
    "constraint_covid19": "en",
    "pheme":              "en",
}
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_int_safe(value) -> Optional[int]:
    """Converte valor para int com segurança."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
# ---------------------------------------------------------------------------
# Validação de schema
# ---------------------------------------------------------------------------
def validate_schema(
    df: pd.DataFrame,
    dataset_name: str = "",
) -> List[str]:
    """
    Verifica colunas obrigatórias e valores de labels.
    Parâmetros
    ----------
    df           : DataFrame carregado pelo loader.
    dataset_name : nome do dataset para mensagens.
    Retorna
    -------
    Lista de strings de erro. Lista vazia = sem erros.
    """
    errors: List[str] = []
    # Verifica colunas obrigatórias
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(
                f"[{dataset_name}] Missing "
                f"required column: '{col}'"
            )
    # Verifica valores de labels
    if "label" in df.columns:
        try:
            unique_labels = set(
                int(v)
                for v in df["label"].dropna().unique()
            )
        except (ValueError, TypeError):
            unique_labels = set(
                df["label"].dropna().unique()
            )
        invalid_labels = unique_labels - VALID_LABELS
        if invalid_labels:
            errors.append(
                f"[{dataset_name}] Unexpected "
                f"label values: {invalid_labels}"
            )
    # Verifica textos vazios
    if "text" in df.columns:
        text_series = df["text"].astype(str)
        empty_texts = int(
            (
                df["text"].isna()
                | (text_series.str.strip() == "")
                | (text_series.str.lower() == "nan")
            ).sum()
        )
        if empty_texts > 0:
            errors.append(
                f"[{dataset_name}] "
                f"{empty_texts} rows with empty text"
            )
    return errors
# ---------------------------------------------------------------------------
# Validação de contagem mínima
# ---------------------------------------------------------------------------
def validate_minimum_samples(
    df: pd.DataFrame,
    dataset_name: str = "",
) -> List[str]:
    """
    Verifica contagem mínima de amostras total e
    por classe.
    Parâmetros
    ----------
    df           : DataFrame carregado pelo loader.
    dataset_name : nome do dataset para mensagens.
    Retorna
    -------
    Lista de strings de aviso. Lista vazia = sem avisos.
    """
    errors: List[str] = []
    min_total = DATASET_MIN_SAMPLES.get(
        dataset_name, DEFAULT_MIN_SAMPLES
    )
    total = len(df)
    if total < min_total:
        errors.append(
            f"[{dataset_name}] Only {total} "
            f"samples; expected >= {min_total}."
        )
        return errors
    if "label" not in df.columns:
        return errors
    min_per_class = max(10, min_total // 4)
    for label_val in VALID_LABELS:
        # CORRIGIDO: operação vetorizada em vez de .apply()
        count = int((df["label"] == label_val).sum())
        if count < min_per_class:
            errors.append(
                f"[{dataset_name}] Class "
                f"{label_val} has only "
                f"{count} samples "
                f"(minimum {min_per_class})."
            )
    return errors
# ---------------------------------------------------------------------------
# Relatório de qualidade
# ---------------------------------------------------------------------------
def print_dataset_report(
    df: pd.DataFrame,
    dataset_name: str = "",
) -> None:
    """
    Imprime relatório de qualidade do dataset
    via logger.
    """
    label_counts: Dict[int, int] = {}
    if "label" in df.columns:
        label_counts = {
            int(k): int(v)
            for k, v in
            df["label"].value_counts().to_dict().items()
        }
    n            = len(df)
    text_lengths = (
        df["text"].str.split().str.len()
        if "text" in df.columns
        else pd.Series(dtype=int)
    )
    logger.info("=" * 60)
    logger.info(
        "Dataset: %s",
        dataset_name or "unknown",
    )
    logger.info("Total records   : %d", n)
    logger.info(
        "Label counts    : %s", label_counts
    )
    if 0 in label_counts and 1 in label_counts:
        ratio = (
            label_counts[1] / label_counts[0]
            if label_counts[0] > 0
            else float("inf")
        )
        logger.info(
            "Fake/Real ratio : %.3f", ratio
        )
    if not text_lengths.empty:
        logger.info(
            "Text length (tokens) — "
            "min: %d | mean: %.1f | "
            "median: %.1f | max: %d",
            int(text_lengths.min()),
            float(text_lengths.mean()),
            float(text_lengths.median()),
            int(text_lengths.max()),
        )
    has_ts = (
        "timestamp" in df.columns
        and df["timestamp"].notna().any()
    )
    logger.info(
        "Has timestamp   : %s", has_ts
    )
    logger.info("=" * 60)
# ---------------------------------------------------------------------------
# Validação combinada
# ---------------------------------------------------------------------------
def assert_valid(
    df: pd.DataFrame,
    dataset_name: str = "",
) -> None:
    """
    Levanta ValueError se schema inválido.
    Loga warnings para contagens baixas.
    Parâmetros
    ----------
    df           : DataFrame carregado pelo loader.
    dataset_name : nome do dataset para mensagens.
    Raises
    ------
    ValueError
        Se validate_schema() retornar erros.
    """
    errors = validate_schema(df, dataset_name)
    if errors:
        raise ValueError("\n".join(errors))
    sample_warnings = validate_minimum_samples(
        df, dataset_name
    )
    for msg in sample_warnings:
        logger.warning(
            "Sample size warning — %s", msg
        )
# ---------------------------------------------------------------------------
# Verificação de disponibilidade de métricas
# ---------------------------------------------------------------------------
def validate_metrics_availability(
    metrics_dir: str,
    seeds: List[int],
    model_names: List[str],
    dataset_names: List[str],
    exp_prefixes: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Valida a existência dos arquivos JSON de métricas
    para cada combinação de experimento, modelo,
    dataset e seed.
    Nomenclatura esperada por experimento:
        EXP_A: EXP_A_{model}_{dataset}_seed{N}.json
        EXP_B: EXP_B_{model}_TRAIN_{train}_TEST_{test}
                   _seed{N}.json
               (apenas pares do mesmo idioma)
        EXP_C: EXP_C_{lang}_{model}_combined_{lang}
                   _test_{dataset}_seed{N}.json
        EXP_D: EXP_D_{model}_{dataset}_TEMPORAL
                   _seed{N}.json
               (apenas datasets com timestamp)

    Parâmetros
    ----------
    metrics_dir   : diretório com arquivos JSON.
    seeds         : lista de seeds usadas.
    model_names   : lista de modelos.
    dataset_names : lista de datasets.
    exp_prefixes  : experimentos a verificar.
                    Padrão: todos (A a F).
    output_path   : caminho para salvar relatório JSON.
    Retorna
    -------
    dict com:
        present_count, missing_count, expected_count,
        complete, missing_files, missing_truncated.
    """
    if exp_prefixes is None:
        exp_prefixes = [
            "EXP_A", "EXP_B", "EXP_C",
            "EXP_D",
        ]
    metrics_path = Path(metrics_dir)
    present: List[str] = []
    missing: List[str] = []
    pt_datasets = {
        "faketruebr", "fakebr", "fakerecogna"
    }
    en_datasets = {
        "fakenewsnet",
        "constraint_covid19",
        "pheme",
    }
    # Importa TEMPORAL_DATASETS de constants
    # Fallback documentado como limitação conhecida:
    # se src.utils.constants estiver desatualizado,
    # datasets temporais adicionados posteriormente
    # não serão verificados no EXP_D.
    try:
        from src.utils.constants import (
            TEMPORAL_DATASETS,
        )
    except ImportError:
        TEMPORAL_DATASETS = frozenset({
            "fakerecogna", "pheme"
        })
        logger.warning(
            "validate_metrics_availability: "
            "src.utils.constants não encontrado. "
            "Usando TEMPORAL_DATASETS fallback: %s. "
            "Se outros datasets temporais existirem, "
            "atualizar src/utils/constants.py.",
            TEMPORAL_DATASETS,
        )
    # Filtra apenas datasets base
    base_dataset_names = [
        ds for ds in dataset_names
        if not ds.startswith("combined_")
    ]
    def check_file(filename: str) -> None:
        fpath = metrics_path / filename
        if (
            fpath.exists()
            and fpath.stat().st_size > 0
        ):
            present.append(filename)
        else:
            missing.append(filename)
    for prefix in exp_prefixes:
        # ── EXP_A ─────────────────────────────────────────
        if prefix == "EXP_A":
            for model in model_names:
                for ds in base_dataset_names:
                    for seed in seeds:
                        check_file(
                            f"EXP_A_{model}"
                            f"_{ds}"
                            f"_seed{seed}.json"
                        )
            continue
        # ── EXP_B ─────────────────────────────────────────
        if prefix == "EXP_B":
            for model in model_names:
                for train_ds in base_dataset_names:
                    for test_ds in base_dataset_names:
                        if train_ds == test_ds:
                            continue
                        # CORRIGIDO: ignora pares
                        # cross-idioma
                        lang_train = DATASET_LANGUAGE.get(
                            train_ds, "unknown"
                        )
                        lang_test  = DATASET_LANGUAGE.get(
                            test_ds, "unknown"
                        )
                        if (
                            lang_train != "unknown"
                            and lang_test != "unknown"
                            and lang_train != lang_test
                        ):
                            logger.debug(
                                "EXP_B: ignorando par "
                                "cross-idioma "
                                "%s (%s) -> %s (%s)",
                                train_ds, lang_train,
                                test_ds, lang_test,
                            )
                            continue
                        for seed in seeds:
                            check_file(
                                f"EXP_B_{model}"
                                f"_TRAIN_{train_ds}"
                                f"_TEST_{test_ds}"
                                f"_seed{seed}.json"
                            )
            continue
        # ── EXP_C ─────────────────────────────────────────
        if prefix == "EXP_C":
            available_pt = sorted(
                set(base_dataset_names) & pt_datasets
            )
            if len(available_pt) >= 2:
                for model in model_names:
                    for test_ds in available_pt:
                        for seed in seeds:
                            check_file(
                                f"EXP_C_pt"
                                f"_{model}"
                                f"_combined_pt"
                                f"_test_{test_ds}"
                                f"_seed{seed}.json"
                            )
            available_en = sorted(
                set(base_dataset_names) & en_datasets
            )
            if len(available_en) >= 2:
                for model in model_names:
                    for test_ds in available_en:
                        for seed in seeds:
                            check_file(
                                f"EXP_C_en"
                                f"_{model}"
                                f"_combined_en"
                                f"_test_{test_ds}"
                                f"_seed{seed}.json"
                            )
            continue
        # ── EXP_D ─────────────────────────────────────────
        if prefix == "EXP_D":
            temporal_available = [
                ds for ds in base_dataset_names
                if ds in TEMPORAL_DATASETS
            ]
            if not temporal_available:
                logger.debug(
                    "EXP_D: nenhum dataset temporal "
                    "disponível — pulando."
                )
                continue
            for model in model_names:
                for ds in temporal_available:
                    for seed in seeds:
                        check_file(
                            f"EXP_D_{model}"
                            f"_{ds}"
                            f"_TEMPORAL"
                            f"_seed{seed}.json"
                        )
            continue

    # ── Relatório ──────────────────────────────────────────
    total_expected        = len(present) + len(missing)
    max_missing_in_report = 100
    report: Dict[str, Any] = {
        "metrics_dir":       str(metrics_dir),
        "seeds":             seeds,
        "models":            model_names,
        "datasets":          dataset_names,
        "exp_prefixes":      exp_prefixes,
        "present_count":     len(present),
        "missing_count":     len(missing),
        "expected_count":    total_expected,
        "complete":          len(missing) == 0,
        "missing_files":     (
            missing[:max_missing_in_report]
        ),
        "missing_truncated": (
            len(missing) > max_missing_in_report
        ),
    }
    # ── Log por experimento ────────────────────────────────
    for prefix in exp_prefixes:
        present_exp = [
            f for f in present
            if f.startswith(f"{prefix}_")
        ]
        missing_exp = [
            f for f in missing
            if f.startswith(f"{prefix}_")
        ]
        total_exp = (
            len(present_exp) + len(missing_exp)
        )
        pct = (
            100 * len(present_exp)
            / max(1, total_exp)
        )
        status = (
            "OK" if not missing_exp else "INCOMPLETO"
        )
        logger.info(
            "%s %s: %d/%d (%.0f%%) | "
            "%d ausente(s)",
            status, prefix,
            len(present_exp),
            total_exp,
            pct,
            len(missing_exp),
        )
        for fname in missing_exp[:3]:
            logger.debug("  AUSENTE: %s", fname)
        if len(missing_exp) > 3:
            logger.debug(
                "  ... +%d não listados",
                len(missing_exp) - 3,
            )
    # ── Log geral ──────────────────────────────────────────
    if missing:
        logger.warning(
            "Results availability: "
            "%d/%d presentes. "
            "Ausentes (%d): %s%s",
            len(present),
            total_expected,
            len(missing),
            str(missing[:3]),
            " ..." if len(missing) > 3 else "",
        )
    else:
        logger.info(
            "Results availability: "
            "todos os %d arquivo(s) "
            "esperados presentes.",
            total_expected,
        )
    # ── Salvar relatório JSON ──────────────────────────────
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(
            parents=True, exist_ok=True
        )
        with open(
            output_file, "w", encoding="utf-8"
        ) as fh:
            json.dump(
                report,
                fh,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(
            "Results availability report saved: %s",
            output_path,
        )
    return report