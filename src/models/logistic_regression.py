# -*- coding: utf-8 -*-
"""
logistic_regression.py — Logistic Regression +
TF-IDF classifier.
Correções aplicadas:
  - Step do classificador renomeado de "classifier"
    para "clf" para consistência com shap_analysis.py.
  - get_feature_importance() e _count_params()
    atualizados para usar "clf".
  - evaluate() verifica _is_fitted explicitamente.
  - _build_pipeline(): comentário sobre achatamento
    dos steps adicionado.
  - load(): log atualizado com nota sobre
    best_params_ e train_time não persistidos.
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from src.features.tfidf import build_tfidf_pipeline
from src.utils.logging import get_logger
from src.utils.metrics import compute_metrics
logger = get_logger(__name__)
class LogisticRegressionModel:
    """
    Logistic Regression com TF-IDF e GridSearchCV
    opcional.
    Pipeline interno:
        tfidf -> clf (LogisticRegression)
    O step é nomeado "clf" para consistência com
    shap_analysis.py e mlp.py.
    """
    def __init__(
        self,
        C: float = 1.0,
        solver: str = "lbfgs",
        class_weight: Optional[str] = "balanced",
        max_iter: int = 1000,
        tfidf_max_features: int = 20000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42,
        param_grid: Optional[Dict[str, Any]] = None,
    ):
        self.C                  = C
        self.solver             = solver
        self.class_weight       = class_weight
        self.max_iter           = max_iter
        self.random_state       = random_state
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range  = tfidf_ngram_range
        self.param_grid         = param_grid
        self.best_params_: Optional[
            Dict[str, Any]
        ] = None
        self.best_score_: Optional[float] = None
        self.cv_results_: Optional[
            Dict[str, Any]
        ] = None
        self.train_time:   float   = 0.0
        self.pipeline:   Optional[Pipeline] = None
        self._is_fitted: bool               = False
    def _build_pipeline(self) -> Pipeline:
        """
        Constrói pipeline TF-IDF + LR.
        Step nomeado "clf" em vez de "classifier"
        para consistência com shap_analysis.py
        e mlp.py.
        """
        tfidf = build_tfidf_pipeline(
            max_features=self.tfidf_max_features,
            ngram_range=self.tfidf_ngram_range,
        )
        lr = LogisticRegression(
            C=self.C,
            solver=self.solver,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        # Achata os steps do pipeline TF-IDF e
        # adiciona o classificador. Se
        # build_tfidf_pipeline() retornar steps
        # adicionais (ex: scaler), eles serão
        # incluídos automaticamente.
        steps = list(tfidf.steps) + [("clf", lr)]
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
    ) -> "LogisticRegressionModel":
        """
        Treina o modelo.
        Parâmetros
        ----------
        X_train    : textos de treino.
        y_train    : labels de treino.
        X_val      : não usado pelo LR — incluído
                     para consistência de API.
        y_val      : não usado pelo LR.
        param_grid : grid de hiperparâmetros.
                     Tem prioridade sobre
                     self.param_grid.
                     Se None e self.param_grid é
                     None, treina sem GridSearch.
        Retorna
        -------
        self
        """
        self.pipeline = self._build_pipeline()
        # param_grid de fit() tem prioridade
        grid  = (
            param_grid
            if param_grid is not None
            else self.param_grid
        )
        start = time.time()
        if grid:
            logger.info(
                "GridSearchCV para "
                "LogisticRegression..."
            )
            logger.info(
                "Parameter grid: %s", grid
            )
            cv = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=self.random_state,
            )
            search = GridSearchCV(
                estimator=self.pipeline,
                param_grid=grid,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
                verbose=1,
                refit=True,
            )
            search.fit(X_train, y_train)
            self.best_params_ = search.best_params_
            self.best_score_  = search.best_score_
            self.cv_results_  = search.cv_results_
            self.pipeline     = (
                search.best_estimator_
            )
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
                "Treinando LogisticRegression "
                "sem GridSearch..."
            )
            self.pipeline.fit(X_train, y_train)
        self.train_time = time.time() - start
        self._is_fitted = True
        logger.info(
            "Treino concluído em %.2fs",
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
        Conta parâmetros do LogisticRegression
        fitted.
        Usa "clf" em vez de "classifier".
        """
        try:
            lr = self.pipeline.named_steps["clf"]
            return int(
                np.prod(lr.coef_.shape)
                + len(lr.intercept_)
            )
        except Exception:
            return -1
    def get_feature_importance(
        self,
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """
        Retorna top-N features por valor absoluto
        do coeficiente.
        Usa "clf" em vez de "classifier".
        """
        if not self._is_fitted:
            raise RuntimeError(
                "get_feature_importance deve ser "
                "chamado após fit()."
            )
        tfidf = self.pipeline.named_steps["tfidf"]
        lr    = self.pipeline.named_steps["clf"]
        feature_names = (
            tfidf.get_feature_names_out()
        )
        coefs = (
            lr.coef_[0]
            if lr.coef_.ndim > 1
            else lr.coef_
        )
        top_idx = (
            np.argsort(np.abs(coefs))[::-1][:top_n]
        )
        return {
            "features": (
                feature_names[top_idx].tolist()
            ),
            "coefficients": (
                coefs[top_idx].tolist()
            ),
        }
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
            "LogisticRegressionModel saved: %s",
            path,
        )
    @classmethod
    def load(
        cls,
        path: str,
    ) -> "LogisticRegressionModel":
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
                f"Modelo não encontrado: {path}"
            )
        model            = cls()
        model.pipeline   = joblib.load(path)
        model._is_fitted = True
        logger.info(
            "LogisticRegressionModel loaded: %s "
            "(best_params_ e train_time não "
            "persistidos — consultar JSON de "
            "métricas para auditoria).",
            path,
        )
        return model