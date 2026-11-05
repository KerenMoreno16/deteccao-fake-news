"""
temporal.py - Experiment D: chronological split
evaluation.
Correções aplicadas:
  - experiment_id definido ANTES de _save_js_divergence()
    (corrige NameError crítico).
  - print() substituídos por logger.info().
  - _js_divergence_unigrams() ajusta vocabulário apenas
    no conjunto de treino (evita vazamento).
  - Documentação da ordem das operações explicitada.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from src.data.splitting import temporal_split
from src.data.cleaning import clean_text
from src.experiments.train import run_experiment
from src.utils.constants import PT_DATASETS
from src.utils.logging import get_logger
from src.utils.metrics import save_metrics
logger = get_logger(__name__)
def run_temporal_experiment(
    dataset_name: str,
    df: pd.DataFrame,
    model_name: str,
    cfg: Dict[str, Any],
    seed: int = 42,
    timestamp_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    save_dir: str = "results/metrics",
    model_save_dir: str = "results/models",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executa um experimento temporal completo:
      1. Filtra registros sem timestamp válido.
      2. Divide cronologicamente (temporal_split).
      3. Calcula divergência JS treino/teste.
      4. Treina e avalia via run_experiment().
    Parâmetros
    ----------
    dataset_name   : nome do dataset.
    df             : DataFrame completo com timestamp.
    model_name     : nome do modelo.
    cfg            : configuração do experimento.
    seed           : seed de reprodutibilidade.
    timestamp_col  : coluna de timestamp.
    train_ratio    : proporção do treino.
    val_ratio      : proporção da validação.
    save_dir       : diretório para JSONs.
    model_save_dir : diretório para checkpoints.
    model_kwargs   : hiperparâmetros fixos (ex.: best_params
                     do GridSearch da seed de tuning). Evita
                     refazer o GridSearch a cada seed.
    Retorna
    -------
    Dict com resultados do experimento ou {} se
    o dataset não tiver timestamps suficientes.
    Notas
    -----
    run_experiment() é o ÚNICO ponto de gravação
    do JSON de métricas. js_divergence é passado
    via extra_meta para ser incluído no mesmo JSON.
    """
    # ── Verificar timestamps ──────────────────────────────
    if (
        timestamp_col not in df.columns
        or df[timestamp_col].isna().all()
    ):
        logger.warning(
            "Dataset '%s' sem timestamps válidos - "
            "pulando experimento temporal.",
            dataset_name,
        )
        return {}
    n_total  = len(df)
    df_valid = df[
        df[timestamp_col].notna()
    ].reset_index(drop=True)
    n_valid  = len(df_valid)
    if n_valid < n_total:
        logger.warning(
            "Dataset '%s': %d/%d exemplos sem "
            "timestamp excluídos.",
            dataset_name,
            n_total - n_valid,
            n_total,
        )
    if n_valid < 50:
        logger.warning(
            "Dataset '%s': apenas %d exemplos com "
            "timestamp - pulando experimento temporal.",
            dataset_name, n_valid,
        )
        return {}
    # CORRIGIDO: experiment_id definido AQUI,
    # antes de qualquer uso
    experiment_id = (
        f"EXP_D_{model_name}"
        f"_{dataset_name}"
        f"_TEMPORAL_seed{seed}"
    )
    logger.info(
        "Temporal experiment: %s (%d exemplos)",
        experiment_id, n_valid,
    )
    # ── Divisão temporal ──────────────────────────────────
    train_df, val_df, test_df = temporal_split(
        df_valid,
        timestamp_col=timestamp_col,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        # label_col="label" é o padrão - explícito para clareza
    )
    # CORRIGIDO: logger em vez de print
    logger.info(
        "Split | seed=%d | "
        "treino=%d | val=%d | teste=%d",
        seed,
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info(
        "Datas treino: [%s -> %s]",
        train_df[timestamp_col].min(),
        train_df[timestamp_col].max(),
    )
    logger.info(
        "Datas teste:  [%s -> %s]",
        test_df[timestamp_col].min(),
        test_df[timestamp_col].max(),
    )
    # ── Fingerprint dos splits ────────────────────────────
    for nome, split in [
        ("train", train_df),
        ("val",   val_df),
        ("test",  test_df),
    ]:
        colunas = [
            c for c in
            ["timestamp", "text", "label"]
            if c in split.columns
        ]
        conteudo = (
            split[colunas]
            .astype(str)
            .to_csv(index=False)
            .encode("utf-8")
        )
        fingerprint = hashlib.sha256(
            conteudo
        ).hexdigest()
        logger.info(
            "SPLIT_CHECK | seed=%d | split=%s | "
            "n=%d | hash=%s",
            seed, nome, len(split), fingerprint,
        )
    # ── JS Divergence ─────────────────────────────────────
    # CORRIGIDO: experiment_id já está definido aqui
    js = _js_divergence_unigrams(
        train_df["text"],
        test_df["text"],
    )
    logger.info(
        "JS divergence (train vs test): %.4f",
        js,
    )
    _save_js_divergence(
        record={
            "experiment_id":        experiment_id,
            "dataset":              dataset_name,
            "model":                model_name,
            "seed":                 seed,
            "split_type":           "temporal",
            "js_divergence_unigrams": js,
            "n_train":              len(train_df),
            "n_validation":         len(val_df),
            "n_test":               len(test_df),
            "n_with_timestamp":     n_valid,
            "n_without_timestamp":  n_total - n_valid,
        },
        output_path=str(
            Path(save_dir)
            / "temporal"
            / "js_divergence.csv"
        ),
    )
    # ── Idioma ────────────────────────────────────────────
    language = (
        "pt" if dataset_name in PT_DATASETS else "en"
    )
    # ── Treino e avaliação ────────────────────────────────
    # run_experiment() é o ÚNICO ponto de gravação
    # do JSON. js_divergence passado via extra_meta.
    result = run_experiment(
        experiment_id=experiment_id,
        model_name=model_name,
        dataset_name=dataset_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cfg=cfg,
        seed=seed,
        split_type="temporal",
        language=language,
        save_dir=save_dir,
        model_save_dir=model_save_dir,
        model_kwargs=model_kwargs,
        extra_meta={
            "js_divergence_unigrams": round(js, 6),
            "n_with_timestamp":       n_valid,
            "n_without_timestamp":    n_total - n_valid,
            "n_total":                n_total,
        },
    )
    # Adiciona js_divergence ao dict em memória
    # para uso do chamador (não re-salva em disco)
    if "metrics" in result:
        result["metrics"]["js_divergence_unigrams"] = js
    return result
def _js_divergence_unigrams(
    texts_a: pd.Series,
    texts_b: pd.Series,
    vocab_size: int = 5000,
) -> float:
    """
    Calcula a divergência Jensen-Shannon entre as
    distribuições de unigramas de dois conjuntos
    de textos.
    Parâmetros
    ----------
    texts_a    : textos do conjunto de treino.
    texts_b    : textos do conjunto de teste.
    vocab_size : tamanho máximo do vocabulário.
    Retorna
    -------
    float - divergência JS em bits (logaritmo de base 2).
    Intervalo: [0, 1].
    0 = distribuições idênticas.
    Notas metodológicas
    -------------------
    CORRIGIDO: vocabulário ajustado apenas em
    texts_a (treino) para evitar vazamento de
    informação do conjunto de teste na representação
    do treino. Tokens do teste não vistos no treino
    recebem contagem zero (cobertos pela suavização).
    Suavização laplaciana aplicada uma única vez
    antes de calcular m = 0.5*(p+q), eliminando
    dupla suavização assimétrica.
    """
    vec = CountVectorizer(max_features=vocab_size)
    # CORRIGIDO: fit apenas no treino
    vec.fit(texts_a.astype(str))
    p = np.asarray(
        vec.transform(
            texts_a.astype(str)
        ).sum(axis=0),
        dtype=float,
    ).flatten()
    q = np.asarray(
        vec.transform(
            texts_b.astype(str)
        ).sum(axis=0),
        dtype=float,
    ).flatten()
    # Normalizar e suavizar (uma única vez)
    p = (p + 1e-10) / (
        p.sum() + 1e-10 * len(p)
    )
    q = (q + 1e-10) / (
        q.sum() + 1e-10 * len(q)
    )
    
    m  = 0.5 * (p + q)
    js = (
        0.5 * entropy(p, m, base=2)
        + 0.5 * entropy(q, m, base=2)
    )
    if not np.isfinite(js):
        logger.warning(
            "_js_divergence_unigrams: "
            "JS divergence não finita (%.4f). "
            "Verificar textos de entrada.",
            js,
        )
        return 0.0
    return float(js)

def _save_js_divergence(
    record: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Salva registro de divergência JS em CSV.
    Atualiza execuções repetidas pelo experiment_id.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([record])
    if path.exists():
        try:
            existing = pd.read_csv(path)
            data = pd.concat(
                [existing, new_row],
                ignore_index=True,
                sort=False,
            )
        except Exception as e:
            logger.warning(
                "_save_js_divergence: erro ao ler "
                "CSV existente '%s': %s. "
                "Criando novo arquivo.",
                path, e,
            )
            data = new_row
    else:
        data = new_row
    # Remove duplicatas pelo experiment_id
    # (mantém execução mais recente)
    data = data.drop_duplicates(
        subset=["experiment_id"],
        keep="last",
    )
    data.to_csv(path, index=False, encoding="utf-8")
    logger.debug(
        "_save_js_divergence: salvo em %s "
        "(%d registros)",
        path, len(data),
    )