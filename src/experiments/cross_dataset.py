"""
cross_dataset.py - Experiment B: train on one dataset,
test on another (zero-shot cross-dataset evaluation).
O modelo é CARREGADO do disco (treinado no EXP_A)
e avaliado no test set de outro dataset.
Nenhum re-treino ocorre neste experimento.
Correções aplicadas:
  - "bert" adicionado ao _LOADERS.
  - run_all_cross_dataset() filtra pares cross-idioma
    por padrão (cross_language=False).
  - Verificação de dimensões de y_pred e y_test.
  - Duplicação de tempo de inferência removida.
  - Log de progresso aprimorado.
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.data.cleaning import clean_text
from src.models.logistic_regression import (
    LogisticRegressionModel,
)
from src.models.mlp import MLPModel
from src.models.bilstm import BiLSTMModel
from src.models.bertimbau import TransformerClassifier
from src.utils.logging import get_logger
from src.utils.metrics import (
    save_metrics,
    bootstrap_confidence_interval,
    compute_metrics,
)
from src.utils.constants import PT_DATASETS, EN_DATASETS
logger = get_logger(__name__)
# Tamanho mínimo do test set para bootstrap confiável
_BOOTSTRAP_MIN_SAMPLES = 30
# CORRIGIDO: "bert" adicionado ao dicionário de loaders
# Verificar se TransformerClassifier.load serve para
# ambos os modelos transformer. Se BERT EN usar classe
# diferente, substituir pela classe correta.
_LOADERS = {
    "logistic_regression": LogisticRegressionModel.load,
    "mlp":                 MLPModel.load,
    "bilstm":              BiLSTMModel.load,
    "bertimbau":           TransformerClassifier.load,
    "bert":                TransformerClassifier.load,
}
_SKLEARN_MODELS = {"logistic_regression", "mlp"}
def _get_dataset_language(dataset_name: str) -> str:
    """
    Retorna o idioma do dataset.
    PT-BR: faketruebr, fakebr, fakerecogna,
           combined_pt-*
    EN   : fakenewsnet, constraint_covid19, pheme,
           combined_en-*
    """
    if dataset_name in PT_DATASETS:
        return "pt"
    if dataset_name in EN_DATASETS:
        return "en"
    if dataset_name.startswith("combined_pt"):
        return "pt"
    if dataset_name.startswith("combined_en"):
        return "en"
    raise ValueError(
        f"Dataset '{dataset_name}' não pertence a "
        f"nenhum grupo de idioma conhecido. "
        f"PT_DATASETS={PT_DATASETS}, "
        f"EN_DATASETS={EN_DATASETS}"
    )
def _clean(
    texts: pd.Series,
    cfg: Dict[str, Any],
    model_name: Optional[str] = None,
) -> List[str]:
    """Aplica limpeza de texto conforme configuração, com caixa por modelo."""
    pp = cfg.get("preprocessing", {})
    return [
        clean_text(
            t,
            lowercase=pp.get("lowercase", True),
            model_type=model_name,
            remove_url=pp.get("remove_urls", True),
            remove_mention=pp.get(
                "remove_mentions", True
            ),
            hashtag_mode=pp.get("hashtag_mode", (
                "normalize"
                if pp.get("normalize_hashtags", True)
                else "keep"
            )),
            emoji_mode="keep",
        )
        for t in texts.astype(str)
    ]
def run_cross_dataset_experiment(
    train_dataset_name: str,
    test_dataset_name: str,
    model_name: str,
    test_df: pd.DataFrame,
    model_path: str,
    cfg: Dict[str, Any],
    seed: int = 42,
    language: Optional[str] = None,
    save_dir: str = "results/metrics",
) -> Dict[str, Any]:
    """
    Carrega modelo treinado no EXP_A e avalia no
    test set de outro dataset.
    Parâmetros
    ----------
    train_dataset_name : dataset usado no treino (EXP_A)
    test_dataset_name  : dataset de teste
    model_name         : nome do modelo
    test_df            : DataFrame de teste
    model_path         : caminho do modelo salvo
    cfg                : configuração do experimento
    seed               : seed da execução
    language           : idioma (inferido se None)
    save_dir           : diretório para métricas JSON
    Notas
    -----
    NÃO re-treina o modelo.
    Apenas carrega do disco e avalia no test set.
    O pipeline de limpeza deve ser idêntico ao usado
    no EXP_A para garantir comparabilidade.
    """
    experiment_id = (
        f"EXP_B_{model_name}"
        f"_TRAIN_{train_dataset_name}"
        f"_TEST_{test_dataset_name}"
        f"_seed{seed}"
    )
    logger.info("Cross-dataset: %s", experiment_id)
    if language is None:
        language = _get_dataset_language(
            train_dataset_name
        )
    logger.info(
        "Cross-dataset | train=%s | test=%s | "
        "idioma=%s | model=%s",
        train_dataset_name,
        test_dataset_name,
        language,
        model_name,
    )
    # ── Verificar modelo disponível ───────────────────────
    if model_name not in _LOADERS:
        raise ValueError(
            f"Modelo desconhecido: '{model_name}'. "
            f"Disponíveis: {list(_LOADERS)}"
        )
    # ── Carregar modelo do disco ──────────────────────────
    loader = _LOADERS[model_name]
    try:
        model = loader(model_path)
    except Exception as e:
        logger.error(
            "Falha ao carregar modelo %s de %s: %s",
            model_name, model_path, e,
        )
        raise
    # ── Preparar dados de teste ───────────────────────────
    X_test = _clean(test_df["text"], cfg, model_name)
    y_test = test_df["label"].values
    n_test = len(y_test)
    logger.info(
        "Test set: %d amostras | "
        "classes: %s",
        n_test,
        dict(
            zip(
                *np.unique(y_test, return_counts=True)
            )
        ),
    )
    # ── Inferência ────────────────────────────────────────
    t0      = time.time()
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    elapsed = time.time() - t0
    # CORRIGIDO: verificação de consistência de dimensões
    if len(y_pred) != n_test:
        raise ValueError(
            f"Inconsistência de dimensões: "
            f"y_pred={len(y_pred)}, "
            f"y_test={n_test} "
            f"para {experiment_id}."
        )
    if y_proba is not None and len(y_proba) != n_test:
        logger.warning(
            "y_proba tem dimensão diferente de y_test "
            "(%d vs %d) para %s. "
            "Métricas baseadas em probabilidade "
            "serão ignoradas.",
            len(y_proba), n_test, experiment_id,
        )
        y_proba = None
    # ── Métricas ──────────────────────────────────────────
    metrics = compute_metrics(
        y_test, y_pred, y_proba,
        label_names=["real", "fake"],
    )
    # CORRIGIDO: apenas uma chave para tempo de inferência
    metrics["inference_time_seconds"] = round(
        elapsed, 4
    )
    # ── Bootstrap CI ──────────────────────────────────────
    if n_test < _BOOTSTRAP_MIN_SAMPLES:
        logger.warning(
            "Bootstrap CI não confiável: "
            "n_test=%d < min=%d para %s. "
            "IC omitido (None).",
            n_test,
            _BOOTSTRAP_MIN_SAMPLES,
            experiment_id,
        )
        ci_mean = float(
            f1_score(
                y_test, y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        )
        ci_lo, ci_hi = None, None
    else:
        ci_mean, ci_lo, ci_hi = (
            bootstrap_confidence_interval(
                y_test,
                y_pred,
                lambda yt, yp: f1_score(
                    yt, yp,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                ),
                n_iterations=cfg.get(
                    "evaluation", {}
                ).get("bootstrap_iterations", 1000),
                seed=seed,
            )
        )
    metrics["f1_macro_ci_mean"] = ci_mean
    metrics["f1_macro_ci_lo"]   = ci_lo
    metrics["f1_macro_ci_hi"]   = ci_hi
    # ── Metadados ─────────────────────────────────────────
    # CORRIGIDO: removida duplicata "inference_time_s"
    meta = {
        "model":              model_name,
        "train_dataset":      train_dataset_name,
        "test_dataset":       test_dataset_name,
        "seed":               seed,
        "test_size":          n_test,
        "split_type":         "cross",
        "language":           language,
        "model_path":         model_path,
        "bootstrap_reliable": (
            n_test >= _BOOTSTRAP_MIN_SAMPLES
        ),
    }
    # ── Salvar métricas ───────────────────────────────────
    save_metrics(
        metrics,
        path=str(
            Path(save_dir) / f"{experiment_id}.json"
        ),
        experiment_id=experiment_id,
        extra_meta=meta,
    )
    logger.info(
        "Cross F1-macro: %.4f  [IC95%% %s – %s] | "
        "seed=%d | bootstrap_reliable=%s",
        ci_mean,
        "n/a" if ci_lo is None else f"{ci_lo:.4f}",
        "n/a" if ci_hi is None else f"{ci_hi:.4f}",
        seed,
        n_test >= _BOOTSTRAP_MIN_SAMPLES,
    )
    return {
        "experiment_id": experiment_id,
        "metrics":       metrics,
    }
def run_all_cross_dataset(
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    model_names: List[str],
    model_root: str,
    cfg: Dict[str, Any],
    seed: int = 42,
    save_dir: str = "results/metrics",
    cross_language: bool = False,
) -> List[Dict[str, Any]]:
    """
    Itera sobre todos os pares (train, test) de datasets
    × modelos para uma seed.
    Parâmetros
    ----------
    datasets       : {dataset_name: {"train": df,
                                      "val": df,
                                      "test": df}}
    model_names    : modelos a avaliar
    model_root     : diretório raiz dos modelos salvos
    cfg            : configuração do experimento
    seed           : seed da execução
    save_dir       : diretório para métricas JSON
    cross_language : se True, inclui pares PT->EN e EN->PT.
                     Padrão False: apenas mesmo idioma.
    Notas
    -----
    O modelo é CARREGADO do disco (salvo pelo EXP_A).
    O fasttext_path já está embutido no modelo salvo.
    """
    results  : List[Dict[str, Any]] = []
    ds_names : List[str]            = list(
        datasets.keys()
    )
    # Pré-calcula idioma de cada dataset
    ds_languages: Dict[str, str] = {}
    for ds in ds_names:
        try:
            ds_languages[ds] = _get_dataset_language(ds)
        except ValueError:
            logger.warning(
                "Idioma desconhecido para '%s'. "
                "Será incluído em todos os pares.",
                ds,
            )
            ds_languages[ds] = "unknown"
    # Filtra pares válidos
    valid_pairs  : List[tuple] = []
    skipped_lang : int         = 0
    for train_name in ds_names:
        for test_name in ds_names:
            if train_name == test_name:
                continue
            lang_train = ds_languages.get(
                train_name, "unknown"
            )
            lang_test  = ds_languages.get(
                test_name, "unknown"
            )
            # CORRIGIDO: filtra pares cross-idioma
            if (
                not cross_language
                and lang_train != "unknown"
                and lang_test  != "unknown"
                and lang_train != lang_test
            ):
                logger.debug(
                    "EXP_B: par cross-idioma ignorado: "
                    "%s (%s) -> %s (%s). "
                    "Use cross_language=True para incluir.",
                    train_name, lang_train,
                    test_name, lang_test,
                )
                skipped_lang += 1
                continue
            valid_pairs.append((train_name, test_name))
    if skipped_lang > 0:
        logger.info(
            "EXP_B: %d par(es) cross-idioma excluído(s) "
            "(cross_language=False).",
            skipped_lang,
        )
    total = len(valid_pairs) * len(model_names)
    done = 0
    skipped_checkpoint = 0
    logger.info(
        "EXP_B - %d experimentos esperados "
        "(seed=%d, cross_language=%s)",
        total, seed, cross_language,
    )
    for train_name, test_name in valid_pairs:
        language = ds_languages.get(
            train_name, "unknown"
        )
        for model_name in model_names:
            # ── Construir caminho do modelo ───────────────
            exp_a_id = (
                f"EXP_A_{model_name}"
                f"_{train_name}_seed{seed}"
            )
            if model_name in _SKLEARN_MODELS:
                model_path = str(
                    Path(model_root)
                    / model_name
                    / train_name
                    / f"{exp_a_id}.pkl"
                )
            else:
                model_path = str(
                    Path(model_root)
                    / model_name
                    / train_name
                    / exp_a_id
                )
            # ── Verificar checkpoint ──────────────────────
            model_path_obj    = Path(model_path)
            checkpoint_exists = (
                model_path_obj.exists()
                or model_path_obj.with_suffix(
                    ".pkl"
                ).exists()
                or (
                    model_path_obj.is_dir()
                    and any(model_path_obj.iterdir())
                )
            )
            skipped_checkpoint: int = 0
            skipped_error:      int = 0
            if not checkpoint_exists:
                logger.warning(
                    "EXP_B ignorado: checkpoint não "
                    "encontrado para "
                    "%s/%s/seed%d -> %s",
                    model_name, train_name,
                    seed, model_path,
                )
                skipped_checkpoint += 1
                continue
            test_df = datasets[test_name]["test"]
            try:
                res = run_cross_dataset_experiment(
                    train_dataset_name=train_name,
                    test_dataset_name=test_name,
                    model_name=model_name,
                    test_df=test_df,
                    model_path=model_path,
                    cfg=cfg,
                    seed=seed,
                    language=language,
                    save_dir=save_dir,
                )
                results.append(res)
                done += 1
            except Exception as e:
                logger.error(
                    "EXP_B ERRO | %s | "
                    "train=%s | test=%s | "
                    "seed=%d: %s",
                    model_name,
                    train_name,
                    test_name,
                    seed, e,
                )
                skipped_error += 1
    logger.info(
        "EXP_B concluído: "
        "%d OK | "
        "%d sem checkpoint | "
        "%d erros | "
        "%d ignorados (idioma) | "
        "total tentados=%d (seed=%d)",
        done,
        skipped_checkpoint,
        skipped_error,
        skipped_lang,
        done + skipped_checkpoint + skipped_error,
        seed,
    )
    return results