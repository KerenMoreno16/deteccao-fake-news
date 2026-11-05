"""
train.py - Unified training entrypoint for all model
paradigms.
Correções aplicadas:
  - "bert" adicionado ao MODEL_REGISTRY.
  - _TRANSFORMER_MODELS substitui verificação
    restrita a "bertimbau" no bloco de seleção
    do modelo transformer.
  - Comentários explicativos sobre cada decisão
    de design.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.data.cleaning import apply_model_casing, clean_text
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
from src.utils.constants import (
    PT_DATASETS,
    EN_DATASETS,
    BERT_MODEL_BY_LANGUAGE,
    FASTTEXT_MODEL_BY_LANGUAGE,
)
logger = get_logger(__name__)
LABEL_NAMES = ["real", "fake"]
# CORRIGIDO: "bert" adicionado ao registry.
# Verificar se TransformerClassifier serve para ambos
# os modelos transformer. Se BERT EN usar classe
# diferente, importar e substituir aqui.
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel,
    "mlp":                 MLPModel,
    "bilstm":              BiLSTMModel,
    "bertimbau":           TransformerClassifier,
    "bert":                TransformerClassifier,
}
# CORRIGIDO: conjunto de modelos transformer para
# substituir verificação restrita a "bertimbau"
_TRANSFORMER_MODELS = {"bertimbau", "bert"}
_BILSTM_YAML_ONLY_KEYS = {
    "fasttext",
    "fasttext_paths",
}
_SKLEARN_MODELS  = {"logistic_regression", "mlp"}
_NO_RANDOM_STATE = {"bilstm", "bertimbau", "bert"}
_BOOTSTRAP_MIN_SAMPLES = 30
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_series(
    series: pd.Series,
    cfg: Dict[str, Any],
    already_clean: bool = False,
    model_name: Optional[str] = None,
) -> List[str]:
    """
    Aplica pipeline de limpeza ao texto, com política de caixa POR MODELO.
    Se already_clean=True, o DataFrame já foi higienizado (preservando a
    caixa) e aqui só se aplica a política de caixa do modelo:
    minúsculas para TF-IDF/BiLSTM; caixa intacta para Transformers
    (BERTimbau é cased).
    """
    pp = cfg.get("preprocessing", {})
    if already_clean:
        return apply_model_casing(
            series.astype(str).tolist(),
            model_name,
            lowercase=pp.get("lowercase", True),
        )
    return [
        clean_text(
            t,
            lowercase=pp.get("lowercase", True),
            model_type=model_name,
            remove_url=pp.get("remove_urls", True),
            remove_mention=pp.get(
                "remove_mentions", True
            ),
            hashtag_mode=pp.get(
                "hashtag_mode",
                (
                    "normalize"
                    if pp.get("normalize_hashtags", True)
                    else "keep"
                ),
            ),
            emoji_mode="keep",
        )
        for t in series.astype(str)
    ]
def _save_training_history(
    model: Any,
    experiment_id: str,
    save_dir: str = "results/training_history",
) -> Optional[str]:
    """
    Salva histórico de treino (loss por época) em CSV
    se o modelo tiver o atributo 'history'.
    """
    history = getattr(model, "history", None)
    # Retorna None se history ausente ou epoch vazio.
    # Não verifica presença de "loss" - o CSV será
    # salvo com as colunas disponíveis em history.
    if not history or not history.get("epoch"):
        return None
    output_dir  = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df  = pd.DataFrame(history)
    output_path = (
        output_dir / f"{experiment_id}_loss.csv"
    )
    history_df.to_csv(output_path, index=False)
    logger.info(
        "Training history saved: %s", output_path
    )
    return str(output_path)
def _apply_mlp_best_params(
    model_cfg: Dict[str, Any],
    best_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Aplica best_params ao model_cfg.
    Suporta MLP e LR com e sem prefixo
    classifier__.
    """
    model_cfg = model_cfg.copy()
    mapping   = {
        "clf__hidden_layer_sizes": (
            "hidden_layer_sizes"
        ),
        "clf__activation":         "activation",
        "clf__learning_rate_init": (
            "learning_rate_init"
        ),
        "clf__C":            "C",
        "clf__solver":       "solver",
        "clf__class_weight": "class_weight",
    }
    for grid_name, model_key in mapping.items():
        if grid_name in best_params:
            model_cfg[model_key] = (
                best_params[grid_name]
            )
        bare = grid_name.replace("clf__", "")
        if bare in best_params:
            model_cfg[model_key] = best_params[bare]
    return model_cfg
def _resolve_fasttext_path(
    language: str,
    cfg: Dict[str, Any],
) -> Optional[str]:
    """
    Resolve o caminho do modelo FastText para o
    idioma especificado.
    Prioridade:
      1. Caminho definido no YAML (cfg['bilstm']
         ['fasttext'][language])
      2. Caminho default de FASTTEXT_MODEL_BY_LANGUAGE
      3. None (treina com embeddings aleatórios)
    """
    project_root = (
        Path(__file__).resolve().parent.parent.parent
    )
    bilstm_cfg   = cfg.get("bilstm", {})
    fasttext_map = bilstm_cfg.get("fasttext", {})
    yaml_path = fasttext_map.get(language)
    if yaml_path:
        resolved = (
            Path(yaml_path)
            if Path(yaml_path).is_absolute()
            else project_root / yaml_path
        )
        if resolved.exists():
            logger.info(
                "FastText do YAML: %s (idioma=%s)",
                resolved, language,
            )
            return str(resolved)
        else:
            logger.warning(
                "FastText do YAML não encontrado: "
                "%s -> %s",
                yaml_path, resolved,
            )
    default_path = FASTTEXT_MODEL_BY_LANGUAGE.get(
        language
    )
    if default_path:
        resolved_default = (
            Path(default_path)
            if Path(default_path).is_absolute()
            else project_root / default_path
        )
        if resolved_default.exists():
            logger.info(
                "FastText default: %s (idioma=%s)",
                resolved_default, language,
            )
            return str(resolved_default)
    logger.warning(
        "FastText não encontrado para idioma='%s'. "
        "BiLSTM treinará com embeddings aleatórios. "
        "YAML='%s', default='%s'.",
        language,
        fasttext_map.get(language, "não definido"),
        FASTTEXT_MODEL_BY_LANGUAGE.get(
            language, "não definido"
        ),
    )
    return None
def _predict_once(
    model: Any,
    X_test: List[str],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Executa predict() e predict_proba() UMA vez.
    Evita dupla inferência que ocorria quando
    _evaluate_on_test() chamava model.evaluate()
    (inferência interna) e depois model.predict()
    para o bootstrap CI.
    Para BERT: 1 inferência em vez de 3,
    reduzindo tempo de ~51s para ~17s por experimento.
    Retorna
    -------
    (y_pred, y_proba) - y_proba pode ser None se
    predict_proba() não estiver disponível.
    """
    y_pred = model.predict(X_test)
    y_proba: Optional[np.ndarray] = None
    try:
        y_proba = model.predict_proba(X_test)
    except (NotImplementedError, AttributeError):
        logger.debug(
            "predict_proba() não disponível para "
            "%s - y_proba=None.",
            type(model).__name__,
        )
    return y_pred, y_proba
def _evaluate_on_test(
    model: Any,
    X_test: List[str],
    y_test: np.ndarray,
    experiment_id: str,
    dataset_name: str,
    model_name: str,
    seed: int,
    cfg: Dict[str, Any],
    save_dir: str,
    extra_meta: Optional[Dict[str, Any]] = None,
    split_type: str = "random",
    train_size: int = 0,
    val_size: int = 0,
    loss_history_path: Optional[str] = None,
    save_metrics_flag: bool = True,
    save_predictions_flag: bool = True,
    skipped_training: bool = False,
) -> Dict[str, Any]:
    """
    Avalia o modelo no test set, calcula métricas,
    bootstrap CI, salva predições e métricas.
    Inferência executada UMA vez via _predict_once()
    para evitar múltiplas chamadas ao modelo.
    Parâmetros
    ----------
    model                : modelo treinado.
    X_test               : textos de teste.
    y_test               : labels verdadeiros.
    experiment_id        : ID do experimento.
    dataset_name         : nome do dataset.
    model_name           : nome do modelo.
    seed                 : seed da execução.
    cfg                  : configuração.
    save_dir             : diretório para métricas.
    extra_meta           : metadados extras.
    split_type           : tipo de split.
    train_size           : tamanho do train set.
    val_size             : tamanho do val set.
    loss_history_path    : caminho do histórico.
    save_metrics_flag    : salvar métricas em disco.
    save_predictions_flag: salvar predições em disco.
    skipped_training     : True se skip_training.
    Retorna
    -------
    dict com experiment_id, metrics, model,
    loss_history_path, prediction_path,
    skipped_training.
    """
    # ── Inferência única ──────────────────────────────────
    t_inf_start     = time.time()
    y_pred, y_proba = _predict_once(model, X_test)
    inference_time  = time.time() - t_inf_start
    # ── Verificação de dimensões ──────────────────────────
    n_test = len(y_test)
    if len(y_pred) != n_test:
        raise ValueError(
            f"Inconsistência: y_pred={len(y_pred)}, "
            f"y_test={n_test} para {experiment_id}."
        )
    if y_proba is not None and len(y_proba) != n_test:
        logger.warning(
            "y_proba (%d) ≠ y_test (%d) para %s. "
            "Métricas de probabilidade ignoradas.",
            len(y_proba), n_test, experiment_id,
        )
        y_proba = None
    # ── Métricas ──────────────────────────────────────────
    metrics = compute_metrics(
        y_test, y_pred, y_proba,
        label_names=LABEL_NAMES,
    )
    metrics["inference_time_seconds"] = round(
        inference_time, 4
    )
    # Tempo de treino do modelo (se disponível)
    train_time = getattr(model, "train_time", None)
    if train_time is not None:
        metrics["train_time_seconds"] = round(
            float(train_time), 2
        )
    # ── Salvar predições ──────────────────────────────────
    prediction_path: Optional[str] = None
    if save_predictions_flag:
        prediction_record = {
            "experiment_id": experiment_id,
            "model":         model_name,
            "dataset":       dataset_name,
            "seed":          seed,
            "y_true":        y_test.tolist(),
            "y_pred":        y_pred.tolist(),
            "y_proba": (
                y_proba.tolist()
                if y_proba is not None
                else None
            ),
        }
        prediction_dir = Path("results/predictions")
        prediction_dir.mkdir(
            parents=True, exist_ok=True
        )
        pred_path = (
            prediction_dir
            / f"{experiment_id}_predictions.json"
        )
        try:
            with open(
                pred_path, "w", encoding="utf-8"
            ) as f:
                json.dump(
                    prediction_record,
                    f,
                    default=str,
                )
            prediction_path = str(pred_path)
            logger.info(
                "Predictions saved: %s",
                prediction_path,
            )
        except OSError as e:
            logger.warning(
                "Falha ao salvar predições %s: %s",
                pred_path, e,
            )
    # ── Bootstrap CI ──────────────────────────────────────
    # O IC mede a incerteza amostral do TESTE para um modelo fixo (não a
    # variância de treino - essa é o desvio-padrão entre seeds).
    # n pequeno → IC = None (sem placeholder [0, 1]).
    if n_test < _BOOTSTRAP_MIN_SAMPLES:
        logger.warning(
            "Bootstrap CI não confiável: "
            "n_test=%d < min=%d para %s. "
            "IC omitido (None).",
            n_test, _BOOTSTRAP_MIN_SAMPLES,
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
                groups=(extra_meta or {}).get("bootstrap_groups"),
            )
        )
    metrics["f1_macro_ci_mean"] = ci_mean
    metrics["f1_macro_ci_lo"]   = ci_lo
    metrics["f1_macro_ci_hi"]   = ci_hi
    # ── Metadados ─────────────────────────────────────────
    meta: Dict[str, Any] = {
        "model":              model_name,
        "dataset":            dataset_name,
        "seed":               seed,
        "train_size":         train_size,
        "val_size":           val_size,
        "test_size":          n_test,
        "split_type":         split_type,
        "skipped_training":   skipped_training,
        "bootstrap_reliable": (
            n_test >= _BOOTSTRAP_MIN_SAMPLES
        ),
        "prediction_path":    prediction_path,
    }
    if loss_history_path is not None:
        meta["loss_history_path"] = loss_history_path
    if extra_meta:
        meta.update(extra_meta)
    # ── Salvar métricas ───────────────────────────────────
    if save_metrics_flag:
        save_metrics(
            metrics,
            path=str(
                Path(save_dir)
                / f"{experiment_id}.json"
            ),
            experiment_id=experiment_id,
            extra_meta=meta,
        )
    logger.info(
        "F1-macro: %.4f [IC95%% %s – %s] | "
        "seed=%d | bootstrap_reliable=%s",
        ci_mean,
        "n/a" if ci_lo is None else f"{ci_lo:.4f}",
        "n/a" if ci_hi is None else f"{ci_hi:.4f}",
        seed,
        n_test >= _BOOTSTRAP_MIN_SAMPLES,
    )
    return {
        "experiment_id":     experiment_id,
        "metrics":           metrics,
        "model":             model,
        "loss_history_path": loss_history_path,
        "prediction_path":   prediction_path,
        "skipped_training":  skipped_training,
    }
# ---------------------------------------------------------------------------
# run_experiment() - entrypoint unificado
# ---------------------------------------------------------------------------
def run_experiment(
    experiment_id: str,
    model_name: str,
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    seed: int = 42,
    model_kwargs: Optional[Dict[str, Any]] = None,
    best_params: Optional[Dict[str, Any]] = None,
    language: Optional[str] = None,
    split_type: str = "random",
    extra_meta: Optional[Dict[str, Any]] = None,
    save_dir: str = "results/metrics",
    model_save_dir: str = "results/models",
    already_clean: bool = True,
    pretrained_model: Optional[Any] = None,
    skip_training: bool = False,
    save_model: bool = True,
    save_metrics_flag: bool = True,
    save_predictions_flag: bool = True,
) -> Dict[str, Any]:
    """
    Entrypoint unificado para treino e avaliação.
    Parâmetros
    ----------
    experiment_id         : identificador único.
    model_name            : chave do MODEL_REGISTRY.
    dataset_name          : nome do dataset.
    train_df              : DataFrame de treino.
    val_df                : DataFrame de validação.
    test_df               : DataFrame de teste.
    cfg                   : configuração YAML.
    seed                  : seed de reprodução.
    model_kwargs          : kwargs extras para modelo.
    best_params           : melhores params GridSearch.
    language              : idioma ('pt' ou 'en').
    split_type            : tipo de split.
    extra_meta            : metadados extras no JSON.
    save_dir              : diretório para métricas.
    model_save_dir        : diretório para modelos.
    already_clean         : pula limpeza se True.
    pretrained_model      : modelo pré-treinado
                            (usado com skip_training).
    skip_training         : pula treino se True.
    save_model            : salva modelo em disco.
    save_metrics_flag     : salva métricas em disco.
    save_predictions_flag : salva predições em disco.
    Retorna
    -------
    dict com:
        experiment_id, metrics, model,
        loss_history_path, prediction_path,
        grid_search_enabled, best_params,
        language, skipped_training, training_time_s.
    Notas
    -----
    Este é o ÚNICO ponto de salvamento de métricas
    e predições no pipeline. Não chamar save_metrics()
    externamente para o mesmo experiment_id.
    """
    logger.info("=" * 60)
    logger.info(
        "EXP: %s | model: %s | dataset: %s | "
        "seed: %d",
        experiment_id, model_name,
        dataset_name, seed,
    )
    # ── Verificar model_name ──────────────────────────────
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo desconhecido: '{model_name}'. "
            f"Disponíveis: {list(MODEL_REGISTRY)}"
        )
    # ── Inferir idioma ────────────────────────────────────
    if language is None:
        if dataset_name in PT_DATASETS:
            language = "pt"
        elif dataset_name in EN_DATASETS:
            language = "en"
        elif dataset_name.startswith("combined_pt"):
            language = "pt"
        elif dataset_name.startswith("combined_en"):
            language = "en"
        else:
            logger.warning(
                "Idioma não inferido para '%s'. "
                "Usando 'pt' como fallback. "
                "Para modelos transformer, será "
                "selecionado: %s",
                dataset_name,
                BERT_MODEL_BY_LANGUAGE.get("pt", "N/D"),
            )
            language = "pt"

    logger.info(
        "Idioma: %s | dataset: %s",
        language, dataset_name,
    )
    # ── Preparar test set ─────────────────────────────────
    X_test = _clean_series(
        test_df["text"], cfg, already_clean, model_name
    )
    y_test = test_df["label"].values
    # ── skip_training ─────────────────────────────────────
    if skip_training and pretrained_model is not None:
        logger.info(
            "skip_training=True: reutilizando modelo "
            "pré-treinado para %s",
            experiment_id,
        )
        return _evaluate_on_test(
            model=pretrained_model,
            X_test=X_test,
            y_test=y_test,
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            model_name=model_name,
            seed=seed,
            cfg=cfg,
            save_dir=save_dir,
            extra_meta=extra_meta,
            split_type=split_type,
            train_size=len(train_df),
            val_size=len(val_df),
            loss_history_path=None,
            save_metrics_flag=save_metrics_flag,
            save_predictions_flag=save_predictions_flag,
            skipped_training=True,
        )
    if skip_training and pretrained_model is None:
        logger.error(
            "skip_training=True mas pretrained_model "
            "é None para %s. "
            "Executando treino normal como fallback.",
            experiment_id,
        )
    # ── Preparar train e val ──────────────────────────────
    X_train = _clean_series(
        train_df["text"], cfg, already_clean, model_name
    )
    X_val = _clean_series(
        val_df["text"], cfg, already_clean, model_name
    )
    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    # ── Configurar modelo ─────────────────────────────────
    ModelClass = MODEL_REGISTRY[model_name]
    model_cfg  = cfg.get(
        model_name.replace("-", "_"), {}
    ).copy()
    if model_kwargs:
        model_cfg.update(model_kwargs)
    # ── Transformer: seleção por idioma ───────────────────
    # CORRIGIDO: _TRANSFORMER_MODELS inclui "bert" e
    # "bertimbau" - evita bloco restrito a "bertimbau"
    if model_name in _TRANSFORMER_MODELS:
        if language not in BERT_MODEL_BY_LANGUAGE:
            raise ValueError(
                f"Idioma inválido para "
                f"'{model_name}': {language!r}. "
                "Use 'pt' ou 'en'."
            )
        selected_model   = (
            BERT_MODEL_BY_LANGUAGE[language]
        )
        configured_model = model_cfg.get("model_name")
        if configured_model != selected_model:
            logger.info(
                "Transformer ajustado: "
                "idioma=%s | modelo=%s | "
                "dataset=%s | model_key=%s",
                language, selected_model,
                dataset_name, model_name,
            )
        model_cfg["model_name"] = selected_model
    # ── BiLSTM: injetar fasttext_path ─────────────────────
    if model_name == "bilstm":
        fasttext_path = _resolve_fasttext_path(
            language, cfg
        )
        if fasttext_path is not None:
            model_cfg["fasttext_path"] = fasttext_path
            logger.info(
                "BiLSTM FastText: %s (idioma=%s)",
                fasttext_path, language,
            )
        else:
            logger.warning(
                "BiLSTM sem embeddings pré-treinados "
                "para dataset='%s' (idioma='%s').",
                dataset_name, language,
            )
        for _yaml_key in _BILSTM_YAML_ONLY_KEYS:
            model_cfg.pop(_yaml_key, None)
    # ── MLP: aplicar best_params ──────────────────────────
    if (
        model_name == "mlp"
        and best_params is not None
    ):
        model_cfg = _apply_mlp_best_params(
            model_cfg, best_params
        )
        model_cfg["skip_grid_search"] = True
        logger.info(
            "MLP com hiperparâmetros fixos: %s",
            best_params,
        )
    # ── Extrair param_grid e skip_grid_search ─────────────
    param_grid            = model_cfg.pop(
        "param_grid", None
    )
    skip_grid_search_flag = model_cfg.pop(
        "skip_grid_search", False
    )
    use_grid_search = (
        model_name in _SKLEARN_MODELS
        and param_grid is not None
        and len(param_grid) > 0
        and not skip_grid_search_flag
    )
    if use_grid_search:
        logger.info(
            "GridSearchCV ATIVO para %s",
            model_name,
        )
    elif model_name in _SKLEARN_MODELS:
        logger.info(
            "GridSearchCV INATIVO para %s",
            model_name,
        )
    # random_state apenas para modelos sklearn
    # BiLSTM e transformers usam seed via fit()
    if model_name not in _NO_RANDOM_STATE:
        model_cfg["random_state"] = seed
    # ── Instanciar modelo ─────────────────────────────────
    model = ModelClass(**model_cfg)
    # ── Treino ────────────────────────────────────────────
    logger.info(
        "Iniciando treino: %s", experiment_id
    )
    t0 = time.time()
    if use_grid_search:
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            param_grid=param_grid,
        )
    else:
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
        )
    t_train_end   = time.time()
    elapsed_train = t_train_end - t0
    # Tempo de treino: preferir atributo do modelo
    # se disponível (mais preciso para BERT/BiLSTM)
    _model_train_time = getattr(
        model, "train_time", None
    )
    if _model_train_time is None:
        _model_train_time = getattr(
            model, "training_time_", None
        )
    # FIX: usar "is not None" em vez de "or"
    # para evitar que 0.0 (falsy) seja ignorado
    train_time_s = (
        float(_model_train_time)
        if _model_train_time is not None
        else elapsed_train
    )
    logger.info(
        "Treino concluído: %s (%.2fs)",
        experiment_id, train_time_s,
    )
    # ── Histórico de treino ───────────────────────────────
    loss_history_path = _save_training_history(
        model, experiment_id
    )
    # ── Salvar modelo ─────────────────────────────────────
    if save_model:
        model_dir = (
            Path(model_save_dir)
            / model_name
            / dataset_name
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = (
            model_dir / f"{experiment_id}.pkl"
            if model_name in _SKLEARN_MODELS
            else model_dir / experiment_id
        )
        try:
            model.save(str(model_path))
            logger.info(
                "Modelo salvo: %s", model_path
            )
        except Exception as e:
            logger.warning(
                "Falha ao salvar modelo %s: %s",
                experiment_id, e,
            )
    # ── Metadados de treino ───────────────────────────────
    grid_meta: Dict[str, Any] = {
        "grid_search_enabled": use_grid_search,
    }
    if use_grid_search:
        grid_meta["param_grid"] = param_grid
        if (
            hasattr(model, "best_params_")
            and model.best_params_ is not None
        ):
            grid_meta["best_params"] = (
                model.best_params_
            )
        if (
            hasattr(model, "best_score_")
            and model.best_score_ is not None
        ):
            grid_meta["best_cv_f1_macro"] = (
                model.best_score_
            )
    if loss_history_path is not None:
        grid_meta["loss_history_path"] = (
            loss_history_path
        )
    merged_meta: Dict[str, Any] = {
        "model":            model_name,
        "dataset":          dataset_name,
        "seed":             seed,
        "train_size":       len(train_df),
        "val_size":         len(val_df),
        "test_size":        len(test_df),
        "split_type":       split_type,
        "language":         language,
        "skipped_training": False,
        "training_time_s":  round(train_time_s, 2),
    }
    merged_meta.update(grid_meta)
    if extra_meta:
        merged_meta.update(extra_meta)
    # Resolver best_params final
    if "best_params" not in merged_meta:
        # FIX: usar "is not None" em vez de "or"
        # para evitar que dicionário vazio {} (falsy)
        # seja ignorado quando GridSearch retorna
        # resultado sem parâmetros variáveis
        _bp_from_model = getattr(
            model, "best_params_", None
        )
        _bp_from_kwargs = None
        if model_kwargs:
            _known_bp_keys = {
                # MLP - sem prefixo
                "hidden_layer_sizes",
                "activation",
                "learning_rate_init",
                # LR - sem prefixo
                "C",
                "solver",
                "class_weight",
                # MLP + LR - com prefixo clf__
                "clf__hidden_layer_sizes",
                "clf__activation",
                "clf__learning_rate_init",
                "clf__C",
                "clf__solver",
                "clf__class_weight",
            }
            _bp_raw = {
                k: v
                for k, v in model_kwargs.items()
                if k in _known_bp_keys
            }
            _bp_from_kwargs = _bp_raw or None
        # Prioridade: modelo > kwargs > parâmetro
        if _bp_from_model is not None:
            merged_meta["best_params"] = (
                _bp_from_model
            )
        elif _bp_from_kwargs is not None:
            merged_meta["best_params"] = (
                _bp_from_kwargs
            )
        elif best_params is not None:
            merged_meta["best_params"] = best_params
        else:
            merged_meta["best_params"] = None
    # ── Avaliar no test set ───────────────────────────────
    result = _evaluate_on_test(
        model=model,
        X_test=X_test,
        y_test=y_test,
        experiment_id=experiment_id,
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,
        cfg=cfg,
        save_dir=save_dir,
        extra_meta=merged_meta,
        split_type=split_type,
        train_size=len(train_df),
        val_size=len(val_df),
        loss_history_path=loss_history_path,
        save_metrics_flag=save_metrics_flag,
        save_predictions_flag=save_predictions_flag,
        skipped_training=False,
    )
    # ── Enriquecer resultado ──────────────────────────────
    result["grid_search_enabled"] = use_grid_search
    result["best_params"]         = (
        merged_meta.get("best_params")
    )
    result["language"]             = language
    result["training_time_s"]      = round(
        train_time_s, 2
    )
    return result