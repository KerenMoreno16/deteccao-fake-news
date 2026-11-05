# -*- coding: utf-8 -*-
"""
mlp.py — Multi-Layer Perceptron + TF-IDF classifier
usando sklearn MLPClassifier.
Correções aplicadas:
  - Step do classificador renomeado de "classifier"
    para "clf" para consistência com
    shap_analysis.py e logistic_regression.py.
  - _count_params() atualizado para usar "clf".
  - evaluate() usa inferência única via predict_proba().
  - fit() documenta explicitamente que X_val não é
    usado pelo MLPClassifier.
  - logger.debug emitido quando X_val é passado
    mas ignorado.
  - load(): log atualizado com nota sobre
    best_params_ e train_time não persistidos.
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.utils.class_weight import (
    compute_sample_weight,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from src.features.tfidf import build_tfidf_pipeline
from src.utils.logging import get_logger
from src.utils.metrics import compute_metrics
logger = get_logger(__name__)
class MLPModel:
    """
    MLP classifier com TF-IDF e MaxAbsScaler.
    Pipeline interno:
        tfidf -> scaler -> clf (MLPClassifier)
    O step é nomeado "clf" para consistência com
    shap_analysis.py e logistic_regression.py.
    Nota sobre validação:
        MLPClassifier usa validation_fraction=0.1
        de X_train para early stopping interno.
        O X_val externo NÃO é usado diretamente.
        Isso é diferente do BiLSTM e BERT que usam
        o X_val externo. Documentado no fit().
    """
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (
            256, 128
        ),
        activation: str = "relu",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        alpha: float = 0.0001,
        tfidf_max_features: int = 20000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42,
        param_grid: Optional[
            Dict[str, Any]
        ] = None,
        skip_grid_search: bool = False,
    ):
        self.hidden_layer_sizes  = hidden_layer_sizes
        self.activation          = activation
        self.learning_rate_init  = learning_rate_init
        self.max_iter            = max_iter
        self.early_stopping      = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change    = n_iter_no_change
        self.alpha               = alpha
        self.tfidf_max_features  = tfidf_max_features
        self.tfidf_ngram_range   = tfidf_ngram_range
        self.random_state        = random_state
        self.param_grid          = param_grid
        self.skip_grid_search    = skip_grid_search
        self.best_params_: Optional[
            Dict[str, Any]
        ] = None
        self.best_score_: Optional[float] = None
        self.cv_results_: Optional[Dict]  = None
        self.pipeline:   Optional[Pipeline] = None
        self._is_fitted: bool               = False
        self.train_time: float              = 0.0
    def _build_pipeline(self) -> Pipeline:
        """
        Constrói pipeline TF-IDF + MaxAbsScaler + MLP.
        Step nomeado "clf" em vez de "classifier"
        para consistência com shap_analysis.py e
        logistic_regression.py.
        """
        tfidf = build_tfidf_pipeline(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
            scale=True,
        )
        mlp = MLPClassifier(
            hidden_layer_sizes=(
                self.hidden_layer_sizes
            ),
            activation=self.activation,
            learning_rate_init=(
                self.learning_rate_init
            ),
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=(
                self.validation_fraction
            ),
            n_iter_no_change=self.n_iter_no_change,
            alpha=self.alpha,
            random_state=self.random_state,
            verbose=False,
        )
        # Achata os steps do pipeline TF-IDF e
        # adiciona o classificador. Se
        # build_tfidf_pipeline() retornar steps
        # adicionais, eles serão incluídos
        # automaticamente.
        steps = list(tfidf.steps) + [("clf", mlp)]
        return Pipeline(steps)
    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: Optional[List[str]] = None,
        y_val: Optional[np.ndarray] = None,
        param_grid: Optional[
            Dict[str, Any]
        ] = None,
    ) -> "MLPModel":
        """
        Treina o modelo MLP.
        Parâmetros
        ----------
        X_train    : textos de treino.
        y_train    : labels de treino.
        X_val      : NÃO USADO pelo MLPClassifier.
                     O early stopping interno usa
                     validation_fraction de X_train.
                     Incluído apenas para consistência
                     de API com BiLSTM e BERT.
        y_val      : NÃO USADO (ver X_val).
        param_grid : grid de hiperparâmetros.
                     Tem prioridade sobre
                     self.param_grid.
        Notas metodológicas
        -------------------
        O conjunto de validação do MLP é gerado
        internamente pelo MLPClassifier
        (validation_fraction de X_train por padrão),
        diferente do BiLSTM e BERT que usam o X_val
        externo. Esta diferença deve ser documentada
        na metodologia da monografia.
        Retorna
        -------
        self
        """
        if X_val is not None:
            logger.debug(
                "MLPModel                .fit: X_val fornecido mas "
                "não usado. MLPClassifier usa "
                "validation_fraction=%.1f de "
                "X_train para early stopping "
                "interno.",
                self.validation_fraction,
            )
        start_time    = time.time()
        self.pipeline = self._build_pipeline()
        # param_grid de fit() tem prioridade
        grid = (
            param_grid
            if param_grid is not None
            else self.param_grid
        )
        if grid and not self.skip_grid_search:
            logger.info(
                "GridSearchCV para MLP..."
            )
            cv = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=self.random_state,
            )
            search = GridSearchCV(
                self.pipeline,
                grid,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
                verbose=1,
                refit=True,
                error_score="raise",
            )
            # Pesos "balanced" por amostra:
            # trata o desbalanceamento como
            # LR/BiLSTM/BERT. A divisão interna
            # de validação do MLP também é
            # ponderada.
            sw = compute_sample_weight(
                "balanced", y_train
            )
            search.fit(
                X_train, y_train,
                clf__sample_weight=sw,
            )
            self.pipeline     = (
                search.best_estimator_
            )
            self.best_params_ = search.best_params_
            self.best_score_  = search.best_score_
            self.cv_results_  = search.cv_results_
            logger.info(
                "Best params: %s",
                self.best_params_,
            )
            logger.info(
                "Best CV F1-macro: %.4f",
                self.best_score_,
            )
        else:
            logger.info(
                "MLP sem GridSearch..."
            )
            sw = compute_sample_weight(
                "balanced", y_train
            )
            self.pipeline.fit(
                X_train, y_train,
                clf__sample_weight=sw,
            )
        self._is_fitted = True
        self.train_time = time.time() - start_time
        logger.info(
            "MLP concluído em %.2fs",
            self.train_time,
        )
        return self
    def predict(
        self,
        X: List[str],
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before "
                "calling predict()."
            )
        return self.pipeline.predict(X)
    def predict_proba(
        self,
        X: List[str],
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before "
                "calling predict_proba()."
            )
        return self.pipeline.predict_proba(X)
    def evaluate(
        self,
        X_test: List[str],
        y_test: np.ndarray,
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Avalia o modelo no conjunto de teste.
        Inferência executada uma única vez via
        predict_proba() para consistência com os
        demais modelos.
        Parâmetros
        ----------
        X_test      : textos de teste.
        y_test      : labels verdadeiros.
        label_names : nomes das classes.
        Retorna
        -------
        dict com métricas completas.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before "
                "calling evaluate()."
            )
        start   = time.time()
        y_proba = self.predict_proba(X_test)
        y_pred  = y_proba.argmax(axis=1)
        inference_time = time.time() - start
        metrics = compute_metrics(
            y_test, y_pred, y_proba, label_names
        )
        metrics["train_time_seconds"]     = (
            self.train_time
        )
        metrics["inference_time_seconds"] = (
            inference_time
        )
        metrics["n_params"] = self._count_params()
        return metrics
    def _count_params(self) -> int:
        """
        Conta parâmetros do MLPClassifier fitted.
        Usa "clf" em vez de "classifier".
        """
        try:
            mlp = self.pipeline.named_steps["clf"]
            n_weights = sum(
                w.size for w in mlp.coefs_
            )
            n_biases = sum(
                b.size for b in mlp.intercepts_
            )
            return n_weights + n_biases
        except Exception:
            return -1
    def save(self, path: str) -> None:
        """
        Salva o pipeline em disco via joblib.
        Nota: apenas o pipeline é salvo.
        Metadados como best_params_ e train_time
        não são persistidos. Para auditoria
        completa, consultar o JSON de métricas
        salvo pelo run_experiment().
        """
        Path(path).parent.mkdir(
            parents=True, exist_ok=True
        )
        joblib.dump(self.pipeline, path)
        logger.info(
            "MLP model saved: %s", path
        )
    @classmethod
    def load(
        cls,
        path: str,
    ) -> "MLPModel":
        """
        Carrega modelo salvo do disco.
        Parâmetros
        ----------
        path : caminho para o arquivo .pkl/.joblib.
        Raises
        ------
        FileNotFoundError
            Se o arquivo não existir.
        Notas
        -----
        best_params_ e train_time não são
        restaurados do disco — apenas o pipeline
        sklearn é persistido. Consultar o JSON
        de métricas salvo por run_experiment()
        para auditoria completa.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"MLP model não encontrado: {path}"
            )
        model            = cls()
        model.pipeline   = joblib.load(path)
        model._is_fitted = True
        logger.info(
            "MLPModel loaded: %s "
            "(best_params_ e train_time não "
            "persistidos — consultar JSON de "
            "métricas para auditoria).",
            path,
        )
        return model