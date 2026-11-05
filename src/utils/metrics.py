# -*- coding: utf-8 -*-
"""
metrics.py — Compute and format all evaluation
metrics with bootstrap CIs.
Correções aplicadas:
  - bootstrap_confidence_interval() documenta
    explicitamente que reamostras são com reposição
    e que y_true e y_pred são reamostrados com o
    mesmo índice.
  - save_metrics() estrutura JSON com extra_meta
    no nível raiz e metrics aninhado, consistente
    com load_all_results() em evaluate.py.
  - aggregate_seed_results() mantém per_seed com
    chave int para consistência interna.
  - metrics_formatted: condição redundante
    stats["n"] == 1 or np.isnan(std) simplificada
    para stats["n"] == 1.
  - save_metrics(): comentário sobre default=str
    adicionado.
"""
import json
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
# Tamanho mínimo do test set para CI confiável
_BOOTSTRAP_MIN_SAMPLES = 30
# Métricas escalares agregadas entre seeds
_SCALAR_METRICS: List[str] = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    # Métricas por classe (0=real, 1=fake) —
    # F1-macro pode mascarar uma classe fraca
    "precision_real",
    "recall_real",
    "f1_real",
    "precision_fake",
    "recall_fake",
    "f1_fake",
    "f1_weighted",
    "precision_weighted",
    "recall_weighted",
    "mcc",
    "roc_auc",
    "pr_auc",
    "train_time_seconds",
    "inference_time_seconds",
    "f1_macro_ci_mean",
    "f1_macro_ci_lo",
    "f1_macro_ci_hi",
]
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calcula suite completa de métricas para
    classificação binária ou multiclasse.
    Parâmetros
    ----------
    y_true      : labels verdadeiros.
    y_pred      : labels preditos.
    y_proba     : probabilidades preditas —
                  shape (n,) para binário ou
                  (n, n_classes) para multiclasse.
    label_names : nomes das classes para o relatório.
    Retorna
    -------
    dict com accuracy, precision_macro,
    recall_macro, f1_macro, f1_weighted, mcc,
    roc_auc, pr_auc, confusion_matrix,
    classification_report.
    Raises
    ------
    ValueError
        Se y_true estiver vazio.
    """
    if len(y_true) == 0:
        raise ValueError(
            "compute_metrics: y_true está vazio. "
            "Verifique o split de teste."
        )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    observed = set(
        np.unique(y_true).tolist()
    ) | set(np.unique(y_pred).tolist())
    # Binário: SEMPRE labels=[0, 1] — evita matriz
    # 1x1 / ValueError quando uma partição contém
    # uma única classe.
    labels = (
        [0, 1]
        if observed <= {0, 1}
        else sorted(observed)
    )
    metrics: Dict[str, Any] = {}
    metrics["n_classes_in_test"] = int(
        len(np.unique(y_true))
    )
    metrics["accuracy"] = float(
        accuracy_score(y_true, y_pred)
    )
    for avg in ("macro", "weighted"):
        metrics[f"precision_{avg}"] = float(
            precision_score(
                y_true, y_pred,
                labels=labels,
                average=avg,
                zero_division=0,
            )
        )
        metrics[f"recall_{avg}"] = float(
            recall_score(
                y_true, y_pred,
                labels=labels,
                average=avg,
                zero_division=0,
            )
        )
        metrics[f"f1_{avg}"] = float(
            f1_score(
                y_true, y_pred,
                labels=labels,
                average=avg,
                zero_division=0,
            )
        )
    # Métricas por classe (apenas binário)
    if labels == [0, 1]:
        p, r, f, sup = (
            precision_recall_fscore_support(
                y_true, y_pred,
                labels=labels,
                zero_division=0,
            )
        )
        for i, nm in enumerate(("real", "fake")):
            metrics[f"precision_{nm}"] = float(p[i])
            metrics[f"recall_{nm}"]    = float(r[i])
            metrics[f"f1_{nm}"]        = float(f[i])
            metrics[f"support_{nm}"]   = int(sup[i])
    metrics["mcc"] = float(
        matthews_corrcoef(y_true, y_pred)
    )
    metrics["confusion_matrix"] = (
        confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist()
    )
    report_names = (
        label_names
        if (
            label_names is not None
            and len(label_names) == len(labels)
        )
        else None
    )
    metrics["classification_report"] = (
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=report_names,
            output_dict=True,
            zero_division=0,
        )
    )
    # ROC-AUC e PR-AUC (apenas binário com y_proba)
    if y_proba is not None:
        proba = (
            y_proba[:, 1]
            if y_proba.ndim == 2
            else y_proba
        )
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, proba)
            )
        except ValueError:
            metrics["roc_auc"] = None
            metrics["roc_auc_note"] = (
                "indefinido: y_true com uma "
                "única classe"
            )
        try:
            metrics["pr_auc"] = float(
                average_precision_score(
                    y_true, proba
                )
            )
        except ValueError:
            metrics["pr_auc"] = None
    return metrics
def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[
        [np.ndarray, np.ndarray], float
    ],
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    min_samples: int = _BOOTSTRAP_MIN_SAMPLES,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Calcula intervalo de confiança bootstrap para
    metric_fn(y_true, y_pred).
    Parâmetros
    ----------
    y_true      : labels verdadeiros.
    y_pred      : labels preditos.
    metric_fn   : callable (y_true, y_pred) -> float.
    n_iterations: número de reamostras bootstrap.
    ci          : largura do IC (0.95 = IC 95%).
    seed        : seed para reprodutibilidade.
    min_samples : tamanho mínimo para IC confiável.
    groups      : se informado (ex.: evento do
                  PHEME), faz bootstrap por CLUSTER:
                  reamostra grupos inteiros,
                  respeitando a correlação
                  intra-grupo (Field & Welsh, 2007).
    Retorna
    -------
    (point_estimate, lower_bound, upper_bound)
    Notas metodológicas
    -------------------
    - O primeiro valor é a ESTIMATIVA PONTUAL
      metric_fn(y_true, y_pred) (não a média das
      reamostras).
    - O IC mede apenas a incerteza amostral do
      conjunto de teste para um modelo FIXO; NÃO
      inclui a variância de treino/inicialização
      (essa é medida pelo desvio-padrão entre
      seeds).
    - Reamostras são COM REPOSIÇÃO (bootstrap
      percentile method).
    - y_true e y_pred são reamostrados com o MESMO
      índice para preservar o pareamento entre
      label verdadeiro e predito.
    - Se n < min_samples: retorna
      (point_estimate, nan, nan) com warning (sem
      placeholder [0, 1]). O campo
      bootstrap_reliable=False deve ser setado nos
      metadados pelo caller.
    - IC calculado via percentis empíricos:
      lower = percentil (1-ci)/2 * 100
      upper = percentil (1+ci)/2 * 100
      Para ci=0.95: lower=2.5%, upper=97.5%.
    Raises
    ------
    ValueError
        Se n < 2.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n      = len(y_true)
    if n < 2:
        raise ValueError(
            f"bootstrap_confidence_interval requer "
            f"pelo menos 2 amostras. Recebido: {n}"
        )
    point_estimate = float(
        metric_fn(y_true, y_pred)
    )
    if n < min_samples:
        warnings.warn(
            f"bootstrap_confidence_interval: "
            f"n_samples={n} < min_samples="
            f"{min_samples}. "
            f"CI estatisticamente não confiável. "
            f"Retornando CI=(nan, nan). Resultados "
            f"com test set pequeno devem ser "
            f"interpretados com cautela.",
            UserWarning,
            stacklevel=2,
        )
        return (
            point_estimate,
            float("nan"),
            float("nan"),
        )
    rng    = np.random.default_rng(seed)
    scores = []
    if groups is not None:
        groups = np.asarray(groups)
        uniq, inverse = np.unique(
            groups, return_inverse=True
        )
        members = [
            np.flatnonzero(inverse == k)
            for k in range(len(uniq))
        ]
        for _ in range(n_iterations):
            picked = rng.integers(
                0, len(uniq), size=len(uniq)
            )
            idx = np.concatenate(
                [members[k] for k in picked]
            )
            scores.append(
                float(
                    metric_fn(
                        y_true[idx], y_pred[idx]
                    )
                )
            )
    else:
        for _ in range(n_iterations):
            # Reamostras COM REPOSIÇÃO com mesmo
            # índice para preservar pareamento
            # y_true/y_pred
            idx = rng.integers(0, n, size=n)
            scores.append(
                float(
                    metric_fn(
                        y_true[idx], y_pred[idx]
                    )
                )
            )
    scores_arr = np.array(scores, dtype=float)
    alpha      = (1.0 - ci) / 2.0
    return (
        point_estimate,
        float(
            np.percentile(
                scores_arr, 100 * alpha
            )
        ),
        float(
            np.percentile(
                scores_arr, 100 * (1 - alpha)
            )
        ),
    )
def generalization_gap(
    f1_intra: float,
    f1_cross: float,
) -> float:
    """
    Calcula o generalization gap.
    Gap positivo indica degradação de desempenho
    em domínio não visto (intra > cross).
    Parâmetros
    ----------
    f1_intra : F1-macro no próprio dataset de treino.
    f1_cross : F1-macro em dataset externo.
    Retorna
    -------
    float arredondado a 6 casas decimais.
    """
    return round(f1_intra - f1_cross, 6)
def save_metrics(
    metrics: Dict[str, Any],
    path: str,
    experiment_id: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persiste métricas em arquivo JSON com metadados.
    Estrutura do JSON salvo:
    {
        "experiment_id": "EXP_A_lr_faketruebr_seed42",
        "model":         "logistic_regression",
        "dataset":       "faketruebr",
        "seed":          42,
        "split_type":    "random",
        ...outros campos de extra_meta...,
        "metrics": {
            "f1_macro":  0.87,
            "accuracy":  0.88,
            ...
        }
    }
    Parâmetros
    ----------
    metrics       : dict de métricas calculadas.
    path          : caminho do arquivo JSON de saída.
    experiment_id : identificador único do
                    experimento.
    extra_meta    : metadados adicionais
                    desempacotados no nível raiz
                    do JSON. Campos esperados:
                    model, dataset, seed,
                    split_type,
                    bootstrap_reliable, etc.
    Notas
    -----
    extra_meta é desempacotado no nível RAIZ do
    JSON, não aninhado. Isso garante que
    load_all_results() em evaluate.py possa acessar
    data.get("model") diretamente.
    Se extra_meta contiver "metrics", será
    sobrescrito pelo dict de métricas calculadas.
    """
    payload = {
        "experiment_id": experiment_id,
        **(extra_meta or {}),
        "metrics": metrics,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(
            payload, f,
            indent=2,
            # default=str converte tipos não
            # serializáveis (ex: np.float32) para
            # string. compute_metrics() já converte
            # arrays para .tolist() antes de chegar aqui
            # — este fallback cobre casos residuais
            # em extra_meta.
            default=str,
        )
def aggregate_seed_results(
    result_files: List[str],
    seeds: List[int],
) -> Dict[str, Any]:
    """
    Agrega métricas de múltiplos arquivos JSON
    por seed.
    Parâmetros
    ----------
    result_files : caminhos para JSONs por seed
                   (mesma ordem que seeds).
    seeds        : lista de seeds.
    Retorna
    -------
    dict com:
        n_runs             : número de seeds
        seeds              : lista de seeds
        per_seed           : {seed_int: metrics_dict}
        metrics            : {metric: {mean, std,
                               n, values}}
        metrics_formatted  : {metric: "mean ± std"}
        f1_macro_per_seed  : {"seed42": float, ...}
                             para testes estatísticos
    Notas
    -----
    - std calculado com ddof=1 (desvio padrão
      amostral) para inferência entre execuções.
    - per_seed usa int como chave para consistência
      interna. _normalize_aggregated_results() em
      statistical_tests.py aceita tanto int quanto
      str como chave.
    - f1_macro_per_seed usa str como chave
      ("seed42") para serialização JSON.
    Raises
    ------
    ValueError
        Se result_files e seeds tiverem tamanhos
        diferentes.
    """
    if len(result_files) != len(seeds):
        raise ValueError(
            f"result_files e seeds devem ter o "
            f"mesmo tamanho. "
            f"Got {len(result_files)} files e "
            f"{len(seeds)} seeds."
        )
    per_seed: Dict[int, Dict[str, Any]] = {}
    for seed, fpath in zip(seeds, result_files):
        try:
            with open(
                fpath, "r", encoding="utf-8"
            ) as f:
                payload = json.load(f)
            per_seed[seed] = payload.get(
                "metrics", {}
            )
        except FileNotFoundError:
            per_seed[seed] = {}
            warnings.warn(
                f"Result file not found for "
                f"seed {seed}: {fpath}"
            )
        except json.JSONDecodeError as exc:
            per_seed[seed] = {}
            warnings.warn(
                f"Malformed JSON for seed "
                f"{seed} — {fpath}: {exc}"
            )
    # Agregar métricas escalares
    agg_metrics: Dict[str, Any] = {}
    for metric in _SCALAR_METRICS:
        values = []
        for s in seeds:
            v = per_seed.get(s, {}).get(metric)
            if v is not None:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    pass
        if not values:
            agg_metrics[metric] = {
                "mean":   None,
                "std":    None,
                "n":      0,
                "values": [],
            }
            continue
        arr  = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        # std=nan quando n=1: distingue "desvio zero
        # confirmado" de "apenas 1 seed executou".
        # As tabelas exibirão "†" em vez de "0.0000",
        # alertando o leitor.
        std  = (
            float(np.std(arr, ddof=1))
            if len(arr) > 1
            else float("nan")
        )
        agg_metrics[metric] = {
            "mean":   mean,
            "std":    std,
            "n":      len(values),
            "values": values,
        }
    # f1_macro por seed para testes estatísticos.
    # Usa str como chave para serialização JSON.
    f1_macro_per_seed: Dict[
        str, Optional[float]
    ] = {}
    for s in seeds:
        v = per_seed.get(s, {}).get("f1_macro")
        f1_macro_per_seed[f"seed{s}"] = (
            float(v) if v is not None else None
        )
    # Formatar para exibição
    metrics_formatted: Dict[str, str] = {}
    for metric, stats in agg_metrics.items():
        if stats["mean"] is None:
            metrics_formatted[metric] = "N/A"
        elif stats["n"] == 1:
            # Apenas 1 seed: desvio não definido
            # (std=nan por design) — sinalizar com
            # dagger para alertar o leitor.
            metrics_formatted[metric] = (
                f"{stats['mean']:.4f} ± †"
            )
        else:
            metrics_formatted[metric] = (
                f"{stats['mean']:.4f} "
                f"± {stats['std']:.4f}"
            )
    return {
        "n_runs":            len(seeds),
        "seeds":             seeds,
        "per_seed":          per_seed,
        "metrics":           agg_metrics,
        "metrics_formatted": metrics_formatted,
        "f1_macro_per_seed": f1_macro_per_seed,
    }
def save_aggregated_results(
    aggregated: Dict[str, Any],
    path: str,
    experiment_prefix: str,
    model: str,
    dataset: str,
) -> None:
    """
    Persiste resultado agregado multi-seed em JSON.
    Parâmetros
    ----------
    aggregated        : output de
                        aggregate_seed_results().
    path              : caminho do arquivo JSON.
    experiment_prefix : base ID
                        (ex: "EXP_A_lr_fakebr").
    model             : nome do modelo.
    dataset           : nome do dataset.
    """
    payload = {
        "experiment_prefix": experiment_prefix,
        "model":             model,
        "dataset":           dataset,
        **aggregated,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(
            payload, f,
            indent=2,
            default=str,
        )