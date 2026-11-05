"""
splitting.py — Reproducible train/validation/test splits
with leakage prevention.

Two split strategies are implemented:
  1. Random stratified split (standard)
  2. Temporal split (chronological — for Experiment D)
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def validate_temporal_split_sanity(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str = "label",
    min_class_samples: int = 5,
    strict: bool = True,
) -> None:
    """
    Valida se todas as partições temporais contêm ambas as classes (0 e 1).
    Evita F1/ROC-AUC degenerados quando uma classe está ausente.

    Com strict=True (padrão) levanta ValueError se alguma partição não tiver
    ambas as classes ou tiver menos de `min_class_samples` da minoritária;
    com strict=False apenas registra warning.
    """
    problems = []
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        if len(split) == 0:
            raise ValueError(f"Split temporal '{name}' está vazio.")
        if label_col in split.columns:
            counts = split[label_col].value_counts()
            if len(counts) < 2:
                problems.append(f"{name}: classe única {counts.to_dict()}")
            elif counts.min() < min_class_samples:
                problems.append(
                    f"{name}: classe minoritária com {int(counts.min())} "
                    f"amostra(s) (< {min_class_samples})"
                )
    if problems:
        msg = "Split temporal inválido — " + "; ".join(problems)
        if strict:
            raise ValueError(msg)
        logger.warning(msg)


def _group_split(
    df: pd.DataFrame,
    frac: float,
    label_col: str,
    group_col: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retorna (idx_resto, idx_fração) respeitando grupos (nenhum grupo cruza partições)."""
    groups = df[group_col].to_numpy()
    y = df[label_col].to_numpy()
    n_splits = int(round(1.0 / frac))
    if n_splits >= 2 and abs(1.0 / frac - n_splits) < 1e-6:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return next(sgkf.split(df, y, groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    return next(gss.split(df, y, groups))


def random_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.10,
    label_col: str = "label",
    seed: int = 42,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified random split into train / validation / test.
    The TF-IDF vectorizer and any fitted preprocessors
    must be fit ONLY on the returned *train* split to
    prevent data leakage.

    Se `group_col` for informado (ex.: id de grupo de near-duplicates, evento),
    o split é estratificado E por grupo: nenhum grupo aparece em duas partições
    (StratifiedGroupKFold).
    """
    if label_col not in df.columns:
        raise ValueError(
            f"Coluna '{label_col}' não encontrada. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )
    if not (0 < test_size < 1):
        raise ValueError(f"test_size deve estar em (0, 1). Recebido: {test_size}")
    if not (0 < val_size < 1):
        raise ValueError(f"val_size deve estar em (0, 1). Recebido: {val_size}")
    if test_size + val_size >= 1.0:
        raise ValueError(
            f"test_size ({test_size}) + val_size ({val_size}) >= 1.0. "
            f"Não sobra dados para treino."
        )

    class_counts = df[label_col].value_counts()
    min_samples = int(class_counts.min())
    if min_samples < 3:
        raise ValueError(
            f"Classe minoritária tem apenas {min_samples} amostras. "
            f"Mínimo necessário para estratificação em 3 splits: 3."
        )

    logger.info(
        "random_split: %d amostras | "
        "test=%.0f%% val=%.0f%% train=%.0f%% | seed=%d",
        len(df),
        test_size * 100,
        val_size * 100,
        (1 - test_size - val_size) * 100,
        seed,
    )

    use_groups = group_col is not None and group_col in df.columns
    if group_col is not None and not use_groups:
        raise ValueError(f"group_col '{group_col}' não encontrada em df.")

    if use_groups:
        df = df.reset_index(drop=True)
        tv_idx, te_idx = _group_split(df, test_size, label_col, group_col, seed)
        train_val, test = df.iloc[tv_idx], df.iloc[te_idx]
    else:
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            stratify=df[label_col],
            random_state=seed,
        )

    adjusted_val_size = val_size / (1.0 - test_size)
    min_train_val = int(train_val[label_col].value_counts().min())
    if min_train_val < 2:
        raise ValueError(
            f"Após divisão treino+val/teste, classe minoritária tem apenas "
            f"{min_train_val} amostra(s) em treino+val."
        )

    if use_groups:
        train_val = train_val.reset_index(drop=True)
        tr_idx, va_idx = _group_split(train_val, adjusted_val_size, label_col, group_col, seed)
        train, val = train_val.iloc[tr_idx], train_val.iloc[va_idx]
    else:
        train, val = train_test_split(
            train_val,
            test_size=adjusted_val_size,
            stratify=train_val[label_col],
            random_state=seed,
        )

    _log_split(train, val, test, label_col=label_col)
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def temporal_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    label_col: str = "label"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: older data -> train; newer -> test.
    """
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Column '{timestamp_col}' not found. "
            "Temporal split requires a timestamp column."
        )
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio deve estar em (0, 1). Recebido: {train_ratio}")
    if not (0 < val_ratio < 1):
        raise ValueError(f"val_ratio deve estar em (0, 1). Recebido: {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) >= 1.0. "
            f"Não sobra dados para teste."
        )

    n_valid_ts = int(df[timestamp_col].notna().sum())
    n_total = len(df)
    if n_valid_ts < n_total:
        n_missing = n_total - n_valid_ts
        logger.warning(
            "temporal_split: %d/%d registros com timestamp "
            "ausente (NaT) posicionados no início do treino.",
            n_missing,
            n_total,
        )

    # mergesort = ordenação estável → desempate determinístico entre timestamps iguais
    df_sorted = df.sort_values(
        by=timestamp_col, na_position="first", kind="mergesort"
    ).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end == 0:
        raise ValueError(f"train_ratio={train_ratio} resulta em split de treino vazio para {n} amostras.")
    if val_end == n:
        raise ValueError(f"train_ratio+val_ratio={train_ratio+val_ratio:.2f} resulta em teste vazio.")

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()

    # Validação de integridade de classes (bloqueante: levanta ValueError)
    validate_temporal_split_sanity(train, val, test, label_col=label_col, strict=True)

    # Fronteira: timestamps idênticos em partições distintas (empate de data)
    for a_name, a, b_name, b in (("train", train, "val", val), ("val", val, "test", test)):
        if len(a) and len(b) and a[timestamp_col].max() == b[timestamp_col].min():
            logger.warning(
                "temporal_split: timestamp %s compartilhado entre '%s' e '%s' (empate na fronteira).",
                a[timestamp_col].max(), a_name, b_name,
            )

    def _ts_range(split: pd.DataFrame) -> str:
        if len(split) == 0:
            return "vazio"
        ts_min = split[timestamp_col].min()
        ts_max = split[timestamp_col].max()
        return f"{ts_min} -> {ts_max}"

    logger.info("Temporal split — dataset completo: [%s]", _ts_range(df_sorted))
    logger.info("  Train (%d amostras): [%s]", len(train), _ts_range(train))
    logger.info("  Val   (%d amostras): [%s]", len(val), _ts_range(val))
    logger.info("  Test  (%d amostras): [%s]", len(test), _ts_range(test))
    _log_split(train, val, test, label_col=label_col)

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _log_split(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, label_col: str = "label", ) -> None:
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        if len(split) == 0:
            logger.warning("  %s: VAZIO (0 amostras)", name)
            continue
        dist = split[label_col].value_counts().to_dict() if label_col in split.columns else {}
        logger.info("  %s: %d amostras | %s", name, len(split), dist)
        if label_col in split.columns and len(dist) >= 2:
            counts = split[label_col].value_counts()
            ratio = counts.max() / counts.min()
            if ratio > 5.0:
                logger.warning(
                    "  %s: desbalanceamento detectado — ratio = %.1f:1 (majoritária=%d, minoritária=%d).",
                    name, ratio, int(counts.max()), int(counts.min()),
                )