"""
tfidf.py — TF-IDF feature extraction pipeline.
Enforces the correct fit-on-train-only rule:
the vectorizer is always fitted on the training
split and then applied (transform-only) to
validation and test splits.
Correções aplicadas:
  - fit_transform(): verifica existência do step
    'tfidf' antes de check_is_fitted.
  - save_pipeline(): impede salvamento de pipeline
    não ajustado por padrão (allow_unfitted=False).
  - get_feature_names(): verifica step antes de
    check_is_fitted.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.validation import check_is_fitted
from src.utils.logging import get_logger
logger = get_logger(__name__)
def build_tfidf_pipeline(
    max_features: int = 20000,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    min_df: int = 2,
    analyzer: str = "word",
    scale: bool = False,
) -> Pipeline:
    """
    Constrói pipeline sklearn com TF-IDF e
    opcionalmente MaxAbsScaler.
    Parâmetros
    ----------
    max_features : vocabulário máximo do TF-IDF.
    ngram_range  : range de n-gramas.
    sublinear_tf : aplica log(1 + tf).
    min_df       : frequência mínima de documento.
    analyzer     : 'word' ou 'char'.
    scale        : adiciona MaxAbsScaler após TF-IDF.
                   Recomendado para MLP, não
                   necessário para LR (regularização
                   própria).
    Retorna
    -------
    Pipeline sklearn não ajustado.
    Notas
    -----
    lowercase=False: cleaning.py já normaliza
    maiúsculas. Se fit_transform() for chamado com
    textos não limpos, tokens distintos por
    capitalização serão tratados como features
    diferentes.
    strip_accents=None: preserva caracteres
    acentuados do português.
    """
    steps = [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=sublinear_tf,
                min_df=min_df,
                analyzer=analyzer,
                strip_accents=None,
                lowercase=False,
            ),
        )
    ]
    if scale:
        steps.append(("scaler", MaxAbsScaler()))
    return Pipeline(steps)
def build_tfidf_pipeline_from_cfg(
    cfg: Dict[str, Any],
    scale: bool = False,
) -> Pipeline:
    """
    Factory que lê parâmetros TF-IDF do YAML.
    Parâmetros
    ----------
    cfg   : configuração do experimento
            (output de load_config()).
    scale : adicionar MaxAbsScaler.
    Retorna
    -------
    Pipeline sklearn não ajustado.
    """
    tfidf_cfg = cfg.get("tfidf", {})
    ngram_raw = tfidf_cfg.get("ngram_range", [1, 2])
    return build_tfidf_pipeline(
        max_features=tfidf_cfg.get(
            "max_features", 20000
        ),
        ngram_range=tuple(ngram_raw),
        sublinear_tf=tfidf_cfg.get(
            "sublinear_tf", True
        ),
        min_df=tfidf_cfg.get("min_df", 2),
        analyzer=tfidf_cfg.get("analyzer", "word"),
        scale=scale,
    )
def fit_transform(
    pipeline: Pipeline,
    train_texts: List[str],
    val_texts: Optional[List[str]] = None,
    test_texts: Optional[List[str]] = None,
) -> Tuple[
    sp.spmatrix,
    Optional[sp.spmatrix],
    Optional[sp.spmatrix],
]:
    """
    Ajusta pipeline em train_texts e transforma
    todos os splits.
    IMPORTANTE: fit_transform() deve ser chamado
    APENAS com textos de treino. Textos de validação
    e teste são apenas transformados (sem re-fit)
    para evitar vazamento de dados.
    Parâmetros
    ----------
    pipeline    : pipeline TF-IDF não ajustado.
    train_texts : textos de treino (fit + transform).
    val_texts   : textos de validação (transform only).
    test_texts  : textos de teste (transform only).
    Retorna
    -------
    (X_train, X_val, X_test) — X_val e X_test são
    None se os splits não forem fornecidos.
    Raises
    ------
    ValueError
        Se train_texts estiver vazio ou se o pipeline
        não tiver step 'tfidf'.
    """
    if not train_texts:
        raise ValueError(
            "fit_transform: train_texts está vazio. "
            "Verifique o split de treino."
        )
    # Verificar existência do step
    if "tfidf" not in pipeline.named_steps:
        raise ValueError(
            "fit_transform: pipeline não possui "
            "step 'tfidf'. "
            f"Steps disponíveis: "
            f"{list(pipeline.named_steps.keys())}"
        )
    # Verificar re-fit inesperado
    try:
        check_is_fitted(
            pipeline.named_steps["tfidf"]
        )
        logger.warning(
            "TF-IDF pipeline já estava fitted — "
            "re-fitting com novos dados. "
            "Verifique se isso é intencional."
        )
    except NotFittedError:
        pass
    logger.info(
        "Fitting TF-IDF em %d textos de treino...",
        len(train_texts),
    )
    X_train = pipeline.fit_transform(train_texts)
    logger.info(
        "TF-IDF vocab: %d features",
        X_train.shape[1],
    )
    # Avisar sobre splits vazios antes de transformar
    if (
        val_texts is not None
        and len(val_texts) == 0
    ):
        logger.warning(
            "fit_transform: val_texts está vazio. "
            "X_val terá zero linhas."
        )
    if (
        test_texts is not None
        and len(test_texts) == 0
    ):
        logger.warning(
            "fit_transform: test_texts está vazio. "
            "X_test terá zero linhas."
        )
    X_val = (
        pipeline.transform(val_texts)
        if val_texts is not None
        else None
    )
    X_test = (
        pipeline.transform(test_texts)
        if test_texts is not None
        else None
    )
    return X_train, X_val, X_test

def get_feature_names(
    pipeline: Pipeline,
) -> np.ndarray:
    """
    Retorna os nomes das features do step TF-IDF.
    Raises
    ------
    ValueError
        Se o pipeline não tiver step 'tfidf'.
    RuntimeError
        Se o pipeline não tiver sido ajustado.
    """
    if "tfidf" not in pipeline.named_steps:
        raise ValueError(
            "get_feature_names: pipeline não possui "
            "step 'tfidf'. "
            f"Steps: "
            f"{list(pipeline.named_steps.keys())}"
        )
    try:
        check_is_fitted(
            pipeline.named_steps["tfidf"]
        )
    except NotFittedError as exc:
        raise RuntimeError(
            "get_feature_names chamado antes de "
            "fit_transform. "
            "Ajuste o pipeline primeiro."
        ) from exc
    return (
        pipeline.named_steps["tfidf"]
        .get_feature_names_out()
    )
def save_pipeline(
    pipeline: Pipeline,
    path: str,
    allow_unfitted: bool = False,
) -> None:
    """
    Persiste o pipeline TF-IDF ajustado em disco.
    Parâmetros
    ----------
    pipeline       : pipeline sklearn ajustado.
    path           : caminho do arquivo (.joblib).
    allow_unfitted : se False (padrão), levanta erro
                     se o pipeline não estiver
                     ajustado.
    Raises
    ------
    RuntimeError
        Se pipeline não ajustado e
        allow_unfitted=False.
    """
    if "tfidf" in pipeline.named_steps:
        try:
            check_is_fitted(
                pipeline.named_steps["tfidf"]
            )
        except NotFittedError:
            if not allow_unfitted:
                raise RuntimeError(
                    "save_pipeline: pipeline não "
                    "foi ajustado. "
                    "Chame fit_transform() antes "
                    "de save_pipeline(). "
                    "Use allow_unfitted=True para "
                    "salvar mesmo assim."
                )
            logger.warning(
                "save_pipeline: salvando pipeline "
                "não ajustado "
                "(allow_unfitted=True)."
            )
    Path(path).parent.mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(pipeline, path)
    logger.info(
        "TF-IDF pipeline salvo: %s", path
    )
def load_pipeline(path: str) -> Pipeline:
    """
    Carrega pipeline TF-IDF previamente salvo.
    Parâmetros
    ----------
    path : caminho para o arquivo salvo.
    Retorna
    -------
    Pipeline sklearn carregado.
    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"TF-IDF pipeline não encontrado: "
            f"{path}"
        )
    pipeline = joblib.load(path)
    logger.info(
        "TF-IDF pipeline carregado: %s", path
    )
    return pipeline