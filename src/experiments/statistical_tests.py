"""
statistical_tests.py - Statistical tests for
comparing classification models.
Unidade de comparação: SEEDS (não datasets).
  - Cada seed é uma execução independente.
  - n = número de seeds (10 por padrão).
  - Score por seed = média do F1-macro sobre
    todos os datasets naquela seed.
  - Matriz: rows=seeds, columns=models.
Correções aplicadas:
  - prepare_seed_matrix() substitui
    prepare_dataset_mean_matrix() como método
    principal.
  - cfg.get("tests", []) com fallback para []
    mesmo quando tests=null no YAML.
  - holm_correction() garante campo
    holm_significant em todos os resultados.
  - run_nemenyi() trata KeyError na matriz
    de p-values.
  - _normalize_aggregated_results() robusto
    para todos os formatos de entrada.
  - run_wilcoxon() emite warning quando n<10
    (p_min >= 0.0625, nunca < alpha=0.05).
  - Nemenyi post-hoc protegido por gate:
    só executa se Friedman for significativo
    (Demšar, 2006).
"""
from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, kruskal
from statsmodels.stats.contingency_tables import mcnemar
from src.utils.logging import get_logger
logger = get_logger(__name__)
# ---------------------------------------------------------------------------
# Matriz principal: seeds como observações
# ---------------------------------------------------------------------------
def prepare_seed_matrix(
    df: pd.DataFrame,
    seeds: List[int],
    metric: str = "f1_macro",
) -> pd.DataFrame:
    """
    Constrói matriz pareada com seeds como
    observações.
    Para cada seed, calcula a média do metric
    sobre todos os datasets disponíveis para
    aquele modelo naquela seed.
    Rows = seeds | Columns = models
    Notas metodológicas
    -------------------
    Esta é a unidade de comparação adotada
    na monografia:
      - n = número de seeds (10)
      - Cada seed é uma execução independente
        com inicialização aleatória diferente
      - Mede variabilidade do processo de
        treinamento, não de datasets
      - Wilcoxon exige n >= 10: p_min = 2/2^10
        = 0,00195 < alpha=0,05 (Demšar, 2006)
    Parâmetros
    ----------
    df     : DataFrame com colunas dataset,
             model e {metric}_seed{N}.
    seeds  : lista de seeds.
    metric : métrica a usar.
    Retorna
    -------
    DataFrame com shape (n_seeds, n_models).
    Vazio se dados insuficientes.
    """
    seed_columns = [
        f"{metric}_seed{s}" for s in seeds
    ]
    missing_cols = [
        c for c in seed_columns
        if c not in df.columns
    ]
    if missing_cols:
        logger.warning(
            "prepare_seed_matrix: "
            "colunas ausentes: %s",
            missing_cols,
        )
        return pd.DataFrame()
    if not {"dataset", "model"}.issubset(
        df.columns
    ):
        logger.warning(
            "prepare_seed_matrix: "
            "colunas 'dataset' e/ou 'model' "
            "ausentes."
        )
        return pd.DataFrame()
    if not seeds:
        logger.warning(
            "prepare_seed_matrix: "
            "lista de seeds vazia."
        )
        return pd.DataFrame()
    work = df[
        ["dataset", "model"] + seed_columns
    ].copy()
    for col in seed_columns:
        work[col] = pd.to_numeric(
            work[col], errors="coerce"
        )
    rows = []
    models = sorted(work["model"].unique())
    for s in seeds:
        seed_col = f"{metric}_seed{s}"
        row: Dict[str, Any] = {"seed": s}
        for model in models:
            model_df = work[
                work["model"] == model
            ]
            # Média sobre todos os datasets
            # para este modelo nesta seed
            values = model_df[seed_col].dropna()
            if values.empty:
                row[model] = np.nan
                logger.debug(
                    "prepare_seed_matrix: "
                    "seed=%d model=%s sem dados.",
                    s, model,
                )
            else:
                row[model] = float(
                    values.mean()
                )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    matrix = pd.DataFrame(rows).set_index("seed")
    # Remover modelos com qualquer NaN
    complete_models = [
        col for col in matrix.columns
        if not matrix[col].isna().any()
    ]
    if len(complete_models) < len(matrix.columns):
        dropped = set(matrix.columns) - set(
            complete_models
        )
        logger.warning(
            "prepare_seed_matrix: "
            "modelos removidos por NaN: %s",
            dropped,
        )
        matrix = matrix[complete_models]
    if matrix.empty or matrix.shape[1] < 2:
        logger.warning(
            "prepare_seed_matrix: "
            "menos de 2 modelos completos."
        )
        return pd.DataFrame()
    logger.info(
        "prepare_seed_matrix: "
        "matriz %d seeds x %d modelos | "
        "modelos=%s",
        matrix.shape[0],
        matrix.shape[1],
        matrix.columns.tolist(),
    )
    return matrix
def prepare_dataset_mean_matrix(
    df: pd.DataFrame,
    seeds: List[int],
    metric: str = "f1_macro",
) -> pd.DataFrame:
    """
    Constrói matriz pareada com datasets como
    observações (método alternativo).
    NOTA: use prepare_seed_matrix() para maior
    poder estatístico. Este método é mantido
    para referência e comparação.
    Rows = datasets | Columns = models
    """
    seed_columns = [
        f"{metric}_seed{s}" for s in seeds
    ]
    missing_cols = [
        c for c in seed_columns
        if c not in df.columns
    ]
    if missing_cols:
        logger.warning(
            "prepare_dataset_mean_matrix: "
            "colunas ausentes: %s",
            missing_cols,
        )
        return pd.DataFrame()
    if not {"dataset", "model"}.issubset(
        df.columns
    ):
        logger.warning(
            "prepare_dataset_mean_matrix: "
            "colunas dataset/model ausentes."
        )
        return pd.DataFrame()
    if not seeds:
        return pd.DataFrame()
    work = df[
        ["dataset", "model"] + seed_columns
    ].copy()
    for col in seed_columns:
        work[col] = pd.to_numeric(
            work[col], errors="coerce"
        )
    complete = work.dropna(
        subset=seed_columns, how="any"
    ).copy()
    if complete.empty:
        logger.warning(
            "prepare_dataset_mean_matrix: "
            "nenhuma linha com todas as seeds."
        )
        return pd.DataFrame()
    complete["score_mean"] = complete[
        seed_columns
    ].mean(axis=1)
    duplicated = complete.duplicated(
        subset=["dataset", "model"], keep=False
    )
    if duplicated.any():
        logger.warning(
            "prepare_dataset_mean_matrix: "
            "pares dataset/model duplicados - "
            "consolidando por média."
        )
    matrix = complete.pivot_table(
        index="dataset",
        columns="model",
        values="score_mean",
        aggfunc="mean",
    )
    matrix = matrix.dropna(axis=0, how="any")
    matrix.index.name = "dataset"
    logger.info(
        "prepare_dataset_mean_matrix: "
        "matriz %d datasets x %d modelos",
        matrix.shape[0],
        matrix.shape[1],
    )
    return matrix
# ---------------------------------------------------------------------------
# Testes estatísticos
# ---------------------------------------------------------------------------
def run_wilcoxon(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    model_a: str,
    model_b: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test para duas amostras
    pareadas (unidade = seed).
    ATENÇÃO sobre o poder: o menor p bilateral exato
    possível é 2/2^n: n=5 -> 0,0625 (NUNCA significativo
    a alpha=0,05, independentemente do efeito); n=10 ->
    0,00195. Por isso o projeto usa 10 seeds. Mesmo com
    n=10, após correção de Holm sobre muitas comparações o
    teste só detecta efeitos grandes e consistentes (todas
    ou quase todas as seeds no mesmo sentido). As seeds
    variam apenas o split/inicialização: a inferência é
    sobre a variabilidade de treino, não sobre a população
    de corpora.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            "Wilcoxon requires paired samples "
            "with the same length."
        )
    if len(a) < 2:
        return {
            "model_a":         model_a,
            "model_b":         model_b,
            "statistic":       None,
            "p_value":         None,
            "significant":     False,
            "holm_significant": False,
            "n":               len(a),
            "error": (
                "Not enough paired observations."
            ),
        }
    
    # Advertência de poder: com n<10 o menor p bilateral exato é
    # 2/2^n.  Para n=5 → p_min=0,0625 > 0,05: nunca significativo.
    # Para n=10 → p_min=0,00195 < 0,05 (Demšar, 2006).
    if len(a) < 10:
        logger.warning(
            "Wilcoxon com n=%d: p_min=%.4f. "
            "Com n<10 o teste NUNCA rejeita H0 a alpha=0.05 "
            "(p_min=2/2^n). Usar no mínimo 10 seeds "
            "(Demšar, 2006, JMLR 7:1-30).",
            len(a), 2.0 / (2 ** len(a)),
        )
    try:
        statistic, p_value = wilcoxon(
            a, b, alternative="two-sided"
        )
        return {
            "model_a":         model_a,
            "model_b":         model_b,
            "statistic":       float(statistic),
            "p_value":         float(p_value),
            "significant":     bool(p_value < alpha),
            "holm_significant": False,  # atualizado por holm_correction()
            "alpha":           alpha,
            "n":               len(a),
        }
    except ValueError as exc:
        return {
            "model_a":         model_a,
            "model_b":         model_b,
            "statistic":       None,
            "p_value":         None,
            "significant":     False,
            "holm_significant": False,
            "n":               len(a),
            "error":           str(exc),
        }
def run_friedman(
    scores: Dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Friedman test para três ou mais modelos
    pareados.
    Unidade = seed (n=10). O Friedman (Demšar, 2006)
    exige idealmente N >= 10 blocos e k >= 5 para a
    aproximação qui-quadrado; com poucos modelos o poder
    é limitado e o resultado deve ser lido com cautela.
    """
    if len(scores) < 3:
        raise ValueError(
            "Friedman test requires at least "
            "three models."
        )
    models  = list(scores.keys())
    arrays  = [
        np.asarray(scores[m], dtype=float)
        for m in models
    ]
    lengths = {len(x) for x in arrays}
    if len(lengths) != 1:
        raise ValueError(
            "All models must have the same "
            "number of paired observations."
        )
    statistic, p_value = friedmanchisquare(*arrays)
    return {
        "models":          models,
        "statistic":       float(statistic),
        "p_value":         float(p_value),
        "significant":     bool(p_value < alpha),
        "alpha":           alpha,
        "n":               len(arrays[0]),
        "unit_of_analysis": "seed",
    }
def run_nemenyi(
    scores: Dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Nemenyi post-hoc test para três ou mais
    modelos pareados.
    """
    try:
        import scikit_posthocs as sp
    except ImportError as exc:
        raise ImportError(
            "Nemenyi requires 'scikit-posthocs'. "
            "Install: pip install scikit-posthocs"
        ) from exc
    models = list(scores.keys())
    if len(models) < 3:
        raise ValueError(
            "Nemenyi test requires at least "
            "three models."
        )
    arrays  = [
        np.asarray(scores[m], dtype=float)
        for m in models
    ]
    lengths = {len(x) for x in arrays}
    if len(lengths) != 1:
        raise ValueError(
            "All models must have the same "
            "number of paired observations."
        )
    n = len(arrays[0])
    if n < 2:
        raise ValueError(
            "Nemenyi test requires at least "
            "two paired observations."
        )
    data     = pd.DataFrame(
        np.column_stack(arrays),
        columns=models,
    )
    p_matrix = sp.posthoc_nemenyi_friedman(data)
    # CORRIGIDO: normalizar índices para nomes
    # dos modelos (scikit_posthocs pode retornar
    # índices inteiros)
    if not all(
        col in p_matrix.columns for col in models
    ):
        p_matrix.columns = models
        p_matrix.index   = models
    return {
        "models":           models,
        "alpha":            alpha,
        "n":                n,
        "unit_of_analysis": "seed",
        "p_values":         p_matrix.to_dict(),
    }
# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------
def run_mcnemar(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    model_a: str,
    model_b: str,
    alpha: float = 0.05,
    exact: bool = False,
) -> Dict[str, Any]:
    """
    McNemar test comparando dois classificadores
    nas mesmas instâncias.
    Nota: não integrado ao dispatcher principal.
    Requer predições por instância salvas em
    results/predictions/.
    """
    y_true   = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    if not (
        len(y_true) == len(y_pred_a)
        == len(y_pred_b)
    ):
        raise ValueError(
            "y_true and both prediction arrays "
            "must have the same length."
        )
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true
    table = np.array([
        [
            np.sum( correct_a &  correct_b),
            np.sum( correct_a & ~correct_b),
        ],
        [
            np.sum(~correct_a &  correct_b),
            np.sum(~correct_a & ~correct_b),
        ],
    ])
    result = mcnemar(
        table,
        exact=exact,
        correction=not exact,
    )
    return {
        "model_a":     model_a,
        "model_b":     model_b,
        "table":       table.tolist(),
        "statistic":   float(result.statistic),
        "p_value":     float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
        "alpha":       alpha,
        "n":           len(y_true),
    }
# ---------------------------------------------------------------------------
# Correção múltipla
# ---------------------------------------------------------------------------
def holm_correction(
    results: List[Dict[str, Any]],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Holm-Bonferroni correction para lista de
    resultados de testes pareados.
    CORRIGIDO: garante que holm_significant está
    presente em TODOS os resultados, incluindo
    os sem p_value - evita KeyError downstream.
    """
    # Garantir holm_significant=False em todos
    # antes de processar
    for r in results:
        if "holm_significant" not in r:
            r["holm_significant"] = False
        if "holm_alpha" not in r:
            r["holm_alpha"] = None
    valid = [
        r for r in results
        if r.get("p_value") is not None
    ]
    if not valid:
        return results
    ordered = sorted(
        valid, key=lambda r: r["p_value"]
    )
    m = len(ordered)
    # Holm (1979) step-down: rejeita enquanto p_(i) <= alpha/(m-i);
    # no primeiro p que falha, ESTE E TODOS OS SEGUINTES não são
    # rejeitados (sem o "break" o procedimento perde o controle do FWER).
    # p_holm = max_{j<=i} min(1, (m-j)*p_(j)) (monotônico não decrescente).
    still_rejecting = True
    running_max     = 0.0
    for i, result in enumerate(ordered):
        adjusted_alpha       = alpha / (m - i)
        result["holm_alpha"] = adjusted_alpha
        running_max = max(
            running_max,
            min(1.0, (m - i) * result["p_value"]),
        )
        # p_value_holm: p-value ajustado de Holm
        # (monotônico não-decrescente).
        # Comparar com alpha para rejeição equivale a verificar holm_significant.
        result["p_value_holm"] = running_max
        if still_rejecting and result["p_value"] <= adjusted_alpha:
            result["holm_significant"] = True
        else:
            still_rejecting            = False
            result["holm_significant"] = False
    return results
# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_statistical_results(
    results: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Salva resultados dos testes estatísticos
    em JSON.
    default=str protege contra tipos
    numpy/pandas não serializáveis.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            results, f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_pairwise_wilcoxon(
    scores: Dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Wilcoxon para cada par de modelos.
    """
    results = []
    for model_a, model_b in combinations(
        scores.keys(), 2
    ):
        result = run_wilcoxon(
            scores[model_a],
            scores[model_b],
            model_a,
            model_b,
            alpha=alpha,
        )
        results.append(result)
    return holm_correction(results, alpha=alpha)
def run_pairwise_mcnemar(
    predictions: Dict[str, Dict[str, Sequence[int]]],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    McNemar para cada par de modelos.
    Estrutura esperada:
        predictions = {
            "lr":  {
                "y_true": [...],
                "y_pred": [...],
            },
            "mlp": {
                "y_true": [...],
                "y_pred": [...],
            },
        }
    Nota: não chamado pelo dispatcher principal.
    """
    results = []
    for model_a, model_b in combinations(
        predictions.keys(), 2
    ):
        a = predictions[model_a]
        b = predictions[model_b]
        if not np.array_equal(
            np.asarray(a["y_true"]),
            np.asarray(b["y_true"]),
        ):
            raise ValueError(
                f"Models {model_a} and {model_b} "
                "do not use the same y_true."
            )
        result = run_mcnemar(
            y_true=a["y_true"],
            y_pred_a=a["y_pred"],
            y_pred_b=b["y_pred"],
            model_a=model_a,
            model_b=model_b,
            alpha=alpha,
        )
        results.append(result)
    return holm_correction(results, alpha=alpha)
# ---------------------------------------------------------------------------
# Normalização de aggregated_results
# ---------------------------------------------------------------------------
def _normalize_aggregated_results(
    aggregated_results: Dict[str, Any],
    seeds: List[int],
    metric: str = "f1_macro",
) -> Dict[str, pd.DataFrame]:
    """
    Normaliza aggregated_results para o formato
    {"A": pd.DataFrame, ...}.
    Suporta três formatos de entrada:
    Formato 1 - já normalizado:
        {"A": pd.DataFrame, ...}
    Formato 2 - JSONs agregados do disco:
        {"EXP_A_lr_fakebr_aggregated": {
            "model": "lr",
            "dataset": "fakebr",
            "per_seed": {
                "42": {"f1_macro": 0.96},
                "43": {"f1_macro": 0.97},
            }
        }}
    Formato 3 - campos diretos:
        {"EXP_A_...": {
            "f1_macro_seed42": 0.96,
            "f1_macro_seed43": 0.97,
        }}
    """
    # Caso 1 - já está no formato correto
    if aggregated_results and all(
        isinstance(v, pd.DataFrame)
        for v in aggregated_results.values()
    ):
        return aggregated_results  # type: ignore
    rows: Dict[str, List[Dict]] = {}
    for key, data in aggregated_results.items():
        if not isinstance(data, dict):
            continue
        # Identificar experimento
        key_upper  = key.upper()
        exp_letter = None
        for letter in [
            "A", "B", "C", "D", "E", "F"
        ]:
            if f"EXP_{letter}_" in key_upper:
                exp_letter = letter
                break
        if exp_letter is None:
            prefix = str(
                data.get(
                    "experiment_prefix", ""
                )
            ).upper()
            for letter in [
                "A", "B", "C", "D", "E", "F"
            ]:
                if f"EXP_{letter}" in prefix:
                    exp_letter = letter
                    break
        if exp_letter is None:
            continue
        model   = data.get("model", "")
        dataset = data.get("dataset", "")
        if not model or not dataset:
            continue
        row: Dict[str, Any] = {
            "model":   model,
            "dataset": dataset,
        }
        found_any = False
        # PRIORIDADE 1 - per_seed aninhado
        # {"42": {"f1_macro": 0.96}, ...}
        per_seed = data.get("per_seed")
        if per_seed and isinstance(
            per_seed, dict
        ):
            for s in seeds:
                sd = per_seed.get(s)
                if sd is None:
                    sd = per_seed.get(str(s))
                score = (
                    sd.get(metric)
                    if sd
                    else None
                )
                row[f"{metric}_seed{s}"] = (
                    float(score)
                    if score is not None
                    else np.nan
                )
                if score is not None:
                    found_any = True
        # PRIORIDADE 2 - f1_macro_per_seed flat
        # {"seed42": 0.96, "42": 0.96, ...}
        if not found_any:
            per_seed_flat = data.get(
                f"{metric}_per_seed"
            )
            if per_seed_flat and isinstance(
                per_seed_flat, dict
            ):
                for s in seeds:
                    score = per_seed_flat.get(s)
                    if score is None:
                        score = per_seed_flat.get(
                            str(s)
                        )
                    if score is None:
                        score = per_seed_flat.get(
                            f"seed{s}"
                        )
                    row[f"{metric}_seed{s}"] = (
                        float(score)
                        if score is not None
                        else np.nan
                    )
                    if score is not None:
                        found_any = True
        # PRIORIDADE 3 - campos diretos
        # "f1_macro_seed42": 0.96
        if not found_any:
            metrics_data = data.get(
                "metrics", data
            )
            for s in seeds:
                seed_key = f"{metric}_seed{s}"
                score = metrics_data.get(seed_key)
                if score is None:
                    score = data.get(seed_key)
                row[f"{metric}_seed{s}"] = (
                    float(score)
                    if score is not None
                    else np.nan
                )
                if score is not None:
                    found_any = True
        if not found_any:
            logger.debug(
                "_normalize: %s/%s - score não "
                "encontrado em nenhum formato.",
                model, dataset,
            )
        if exp_letter not in rows:
            rows[exp_letter] = []
        rows[exp_letter].append(row)
    # Converter para DataFrames
    result: Dict[str, pd.DataFrame] = {}
    for exp_letter, row_list in rows.items():
        df = pd.DataFrame(row_list)
        seed_cols = [
            c for c in df.columns
            if c.startswith(f"{metric}_seed")
        ]
        if df.empty or not seed_cols:
            logger.warning(
                "_normalize: EXP_%s sem "
                "colunas de seed.",
                exp_letter,
            )
            continue
        nan_counts = df[seed_cols].isna().sum()
        if nan_counts.any():
            logger.warning(
                "_normalize: EXP_%s NaN: %s",
                exp_letter,
                nan_counts[
                    nan_counts > 0
                ].to_dict(),
            )
        result[exp_letter] = df
        logger.info(
            "_normalize: EXP_%s -> "
            "%d linhas | NaN=%d | seed_cols=%s",
            exp_letter,
            len(df),
            df[seed_cols].isna().sum().sum(),
            seed_cols,
        )
    if not result:
        logger.warning(
            "_normalize: nenhum dado extraído."
        )
    return result

def run_kruskal_wallis_per_dataset(
    df: pd.DataFrame,
    seeds: List[int],
    metric: str = "f1_macro",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Kruskal-Wallis por dataset.
    Para cada dataset, compara os quatro modelos
    usando os vetores de F1-macro das dez seeds
    como observações de cada grupo.
    Quando significativo, aplica Wilcoxon pareado
    com correção de Holm para identificar os pares
    diferentes.
    Retorna dict com resultados por dataset.
    """
    
    seed_columns = [
        f"{metric}_seed{s}" for s in seeds
    ]
    missing = [
        c for c in seed_columns
        if c not in df.columns
    ]
    if missing:
        logger.warning(
            "run_kruskal_wallis_per_dataset: "
            "colunas ausentes: %s", missing,
        )
        return {}
    results: Dict[str, Any] = {}
    datasets = sorted(df["dataset"].unique())
    for dataset in datasets:
        df_ds = df[df["dataset"] == dataset]
        models = sorted(df_ds["model"].unique())
        if len(models) < 2:
            results[dataset] = {
                "skipped": True,
                "reason": "menos de 2 modelos",
            }
            continue
        # Vetor de scores por modelo (seeds como obs)
        scores: Dict[str, np.ndarray] = {}
        for model in models:
            df_m = df_ds[df_ds["model"] == model]
            vals = (
                df_m[seed_columns]
                .values.flatten()
            )
            vals = vals[~np.isnan(vals)]
            if len(vals) == len(seeds):
                scores[model] = vals
            else:
                logger.warning(
                    "KW | %s | %s: %d obs "
                    "(esperado %d) - pulando modelo.",
                    dataset, model,
                    len(vals), len(seeds),
                )
        if len(scores) < 2:
            results[dataset] = {
                "skipped": True,
                "reason": (
                    "modelos insuficientes "
                    "com dados completos"
                ),
            }
            continue
        try:
            stat, p = kruskal(*scores.values())
        except Exception as exc:
            results[dataset] = {
                "error": str(exc),
            }
            continue
        kw_result: Dict[str, Any] = {
            "statistic":        float(stat),
            "p_value":          float(p),
            "significant":      bool(p < alpha),
            "alpha":            alpha,
            "n_per_model":      len(seeds),
            "models":           list(scores.keys()),
            "unit_of_analysis": "seed",
        }
        # Pós-teste Wilcoxon + Holm se significativo
        if p < alpha and len(scores) >= 2:
            pairwise = run_pairwise_wilcoxon(
                scores, alpha=alpha
            )
            kw_result["posthoc_wilcoxon"] = (
                pairwise
            )
            sig_pairs = [
                r for r in pairwise
                if r.get("holm_significant", False)
            ]
            kw_result["significant_pairs"] = (
                sig_pairs
            )
            logger.info(
                "KW | %s: p=%.4g sig=True | "
                "%d pares sig. (Holm)",
                dataset, p, len(sig_pairs),
            )
        else:
            kw_result["posthoc_wilcoxon"] = []
            kw_result["significant_pairs"] = []
            logger.info(
                "KW | %s: p=%.4g sig=False",
                dataset, p,
            )
        results[dataset] = kw_result
    return results

# ---------------------------------------------------------------------------
# Dispatcher principal
# ---------------------------------------------------------------------------
def run_statistical_tests_from_aggregation(
    aggregated_results: Dict[str, Any],
    seeds: List[int],
    cfg: Dict[str, Any],
    output_dir: str = (
        "results/metrics/statistical_tests"
    ),
) -> Dict[str, Any]:
    """
    Executa testes estatísticos usando seeds
    como unidade de comparação.
    Unidade de análise: seeds (n = len(seeds))
    Matriz: rows=seeds, columns=models
    Score por célula: média do F1-macro sobre
    todos os datasets para aquele modelo naquela
    seed.
    Testes disponíveis (configurar em YAML):
        statistical_tests:
          alpha: 0.05
          tests:
            - friedman
            - wilcoxon
            - nemenyi
    """
    statistical_cfg = cfg.get(
        "statistical_tests", {}
    )
    alpha = float(
        statistical_cfg.get("alpha", 0.05)
    )
    # CORRIGIDO: fallback para [] mesmo quando
    # tests=null no YAML (evita TypeError)
    tests_raw = statistical_cfg.get("tests")
    if tests_raw is None:
        tests_raw = []
    enabled = {
        str(test).strip().lower()
        for test in tests_raw
    }
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(
        parents=True, exist_ok=True
    )
    results: Dict[str, Any] = {
        "alpha":            alpha,
        "tests_requested":  sorted(enabled),
        "unit_of_analysis": "seed",
        "n_observations":   len(seeds),
        "seeds":            list(seeds),
    }
    # Normalizar para {"A": pd.DataFrame, ...}
    normalized = _normalize_aggregated_results(
        aggregated_results, seeds
    )
    if not normalized:
        logger.error(
            "run_statistical_tests: "
            "aggregated_results vazio ou formato "
            "não reconhecido. "
            "Chaves recebidas: %s",
            list(aggregated_results.keys())[:5],
        )
        return results
    # Experiment A como base
    df_a = normalized.get("A")
    if df_a is None or (
        isinstance(df_a, pd.DataFrame)
        and df_a.empty
    ):
        logger.warning(
            "No Experiment A data available. "
            "Chaves disponíveis: %s",
            list(normalized.keys()),
        )
        return results
    logger.info(
        "Experiment A DataFrame: "
        "%d linhas x %d colunas",
        len(df_a), len(df_a.columns),
    )
    # ─────────────────────────────────────────
    # Construir matriz seeds × models
    # ─────────────────────────────────────────
    matrix = prepare_seed_matrix(
        df_a,
        seeds=seeds,
        metric="f1_macro",
    )
    if matrix.empty:
        logger.warning(
            "Could not build seed-level "
            "statistical matrix."
        )
        return results
    logger.info(
        "Seed-level matrix: "
        "%d seeds x %d models | models=%s",
        matrix.shape[0],
        matrix.shape[1],
        matrix.columns.tolist(),
    )
    results["matrix"] = {
        "models":             (
            matrix.columns.tolist()
        ),
        "n_observations":     int(len(matrix)),
        "unit_of_analysis":   "seed",
        "metric":             "f1_macro",
        "seed_aggregation":   (
            "mean_over_datasets"
        ),
        "seeds":              list(seeds),
        "index_names":        [
            str(n) for n in matrix.index.names
        ],
        "seeds_included":     [
            int(s) for s in matrix.index.tolist()
        ],
        "datasets_note": (
            "Score por célula = média do "
            "F1-macro sobre todos os datasets "
            "disponíveis para aquele modelo "
            "naquela seed."
        ),
    }
    # Também construir matriz de datasets para
    # referência (não usada nos testes)
    matrix_ds = prepare_dataset_mean_matrix(
        df_a, seeds=seeds, metric="f1_macro"
    )
    if not matrix_ds.empty:
        results["matrix_datasets"] = {
            "models":           (
                matrix_ds.columns.tolist()
            ),
            "n_observations":   int(
                len(matrix_ds)
            ),
            "unit_of_analysis": "dataset",
            "note": (
                "Referência apenas. "
                "Testes usam matriz de seeds."
            ),
            "datasets_included": [
                str(d)
                for d in matrix_ds.index.tolist()
            ],
        }
    scores = {
        model: matrix[model].to_numpy()
        for model in matrix.columns
    }
    # ─────────────────────────────────────────
    # Kruskal-Wallis por dataset
    # ─────────────────────────────────────────
    if "kruskal_wallis" in enabled:
        try:
            kw_results = run_kruskal_wallis_per_dataset(
                df_a,
                seeds=seeds,
                metric="f1_macro",
                alpha=alpha
            )
            results["kruskal_wallis"] = kw_results
            # Salvar CSV resumo por dataset
            kw_rows = []
            for dataset, kw in kw_results.items():
                if kw.get("skipped") or kw.get("error"):
                    continue
                sig_pairs = kw.get(
                    "significant_pairs", []
                )
                kw_rows.append({
                    "dataset":     dataset,
                    "statistic":   kw.get("statistic"),
                    "p_value":     kw.get("p_value"),
                    "significant": kw.get("significant"),
                    "n_sig_pairs": len(sig_pairs),
                })
            if kw_rows:
                kw_df = pd.DataFrame(kw_rows)
                kw_path = (
                    output_dir_path
                    / "kruskal_wallis_per_dataset.csv"
                )
                kw_df.to_csv(kw_path, index=False)
                logger.info(
                    "Kruskal-Wallis: %d datasets | "
                    "%d significativos | "
                    "CSV: %s",
                    len(kw_rows),
                    kw_df["significant"].sum(),
                    kw_path.name,
                )
        except Exception as exc:
            logger.exception(
                "Kruskal-Wallis failed: %s", exc
            )
            results["kruskal_wallis"] = {
                "error": str(exc),
            }
    # ─────────────────────────────────────────
    # Friedman
    # ─────────────────────────────────────────
    if (
        "friedman" in enabled
        or "nemenyi" in enabled
    ):
        if matrix.shape[1] < 3:
            results["friedman"] = {
                "skipped":        True,
                "reason": (
                    "Friedman requires at least "
                    "three models."
                ),
                "n_models":       int(
                    matrix.shape[1]
                ),
                "n_observations": int(
                    len(matrix)
                ),
                "unit_of_analysis": "seed",
            }
            logger.warning(
                "Friedman skipped: "
                "only %d model(s).",
                matrix.shape[1],
            )
        else:
            try:
                friedman_result = run_friedman(
                    scores, alpha=alpha
                )
                results["friedman"] = (
                    friedman_result
                )
                logger.info(
                    "Friedman: "
                    "statistic=%.6f p=%.6g "
                    "significant=%s "
                    "n=%d seeds",
                    friedman_result["statistic"],
                    friedman_result["p_value"],
                    friedman_result["significant"],
                    friedman_result["n"],
                )
            except Exception as exc:
                logger.exception(
                    "Friedman failed: %s", exc
                )
                results["friedman"] = {
                    "error":          str(exc),
                    "models":         (
                        matrix.columns.tolist()
                    ),
                    "n_observations": int(
                        len(matrix)
                    ),
                    "unit_of_analysis": "seed",
                }
    # ─────────────────────────────────────────
    # Wilcoxon pairwise
    # ─────────────────────────────────────────
    if "wilcoxon" in enabled:
        wilcoxon_results = []
        for model_a, model_b in combinations(
            matrix.columns, 2
        ):
            try:
                result = run_wilcoxon(
                    matrix[model_a].to_numpy(),
                    matrix[model_b].to_numpy(),
                    model_a,
                    model_b,
                    alpha=alpha,
                )
            except Exception as exc:
                logger.exception(
                    "Wilcoxon failed for "
                    "%s vs %s: %s",
                    model_a, model_b, exc,
                )
                result = {
                    "model_a":          model_a,
                    "model_b":          model_b,
                    "statistic":        None,
                    "p_value":          None,
                    "significant":      False,
                    "holm_significant": False,
                    "alpha":            alpha,
                    "n":                int(
                        len(matrix)
                    ),
                    "unit_of_analysis": "seed",
                    "error":            str(exc),
                }
            wilcoxon_results.append(result)
        corrected = holm_correction(
            wilcoxon_results, alpha=alpha
        )
        results["wilcoxon"] = corrected
        sig_count = sum(
            1 for r in corrected
            if r.get("holm_significant", False)
        )
        logger.info(
            "Wilcoxon: %d pares | "
            "%d significativos (Holm) | "
            "n=%d seeds",
            len(corrected),
            sig_count,
            len(matrix),
        )
    # ─────────────────────────────────────────
    # Nemenyi post-hoc
    # Gate: só executa se Friedman rejeitou H₀
    # (Demšar, 2006, Seção 3.1: "If the Friedman
    # test is significant, we can proceed with a
    # post-hoc test").
    # ─────────────────────────────────────────
    if "nemenyi" in enabled:
        friedman_significant = results.get(
            "friedman", {}
        ).get("significant", False)
        friedman_was_run = (
            "friedman" in results
            and not results["friedman"].get("skipped", False)
            and not results["friedman"].get("error")
        )
        if friedman_was_run and not friedman_significant:
            results["nemenyi"] = {
                "skipped": True,
                "reason": (
                    "Friedman não significativo - Nemenyi post-hoc "
                    "não aplicado (Demšar, 2006, JMLR 7:1-30). "
                    "Reportar apenas estatísticas descritivas."
                ),
                "friedman_p_value": results["friedman"].get("p_value"),
                "unit_of_analysis": "seed",
            }
            logger.info(
                "Nemenyi skipped: Friedman não significativo "
                "(p=%.4g). Post-hoc não aplicado (Demšar, 2006).",
                results["friedman"].get("p_value", float("nan")),
            )
        elif matrix.shape[1] < 3:
            results["nemenyi"] = {
                "skipped":        True,
                "reason": (
                    "Nemenyi requires at least "
                    "three models."
                ),
                "n_models":       int(
                    matrix.shape[1]
                ),
                "n_observations": int(
                    len(matrix)
                ),
                "unit_of_analysis": "seed",
            }
            logger.warning(
                "Nemenyi skipped: "
                "only %d model(s).",
                matrix.shape[1],
            )
        else:
            try:
                nemenyi_result = run_nemenyi(
                    scores, alpha=alpha
                )
                friedman_info = results.get(
                    "friedman", {}
                )
                nemenyi_result["friedman"] = {
                    "available":   bool(
                        friedman_info
                    ),
                    "significant": friedman_info.get(
                        "significant"
                    ),
                    "p_value":     friedman_info.get(
                        "p_value"
                    ),
                }
                # Pares significativos
                significant_pairs = []
                p_values = (
                    nemenyi_result["p_values"]
                )
                models_n = (
                    nemenyi_result["models"]
                )
                for i, model_a in enumerate(
                    models_n
                ):
                    for model_b in (
                        models_n[i + 1:]
                    ):
                        # CORRIGIDO: try ambas
                        # as ordens para evitar
                        # KeyError
                        try:
                            p_value = float(
                                p_values[model_a][
                                    model_b
                                ]
                            )
                        except KeyError:
                            try:
                                p_value = float(
                                    p_values[
                                        model_b
                                    ][model_a]
                                )
                            except KeyError:
                                logger.warning(
                                    "Nemenyi: "
                                    "p_value não "
                                    "encontrado "
                                    "para %s vs %s",
                                    model_a,
                                    model_b,
                                )
                                continue
                        if p_value < alpha:
                            significant_pairs.append({
                                "model_a":     model_a,
                                "model_b":     model_b,
                                "p_value":     p_value,
                                "alpha":       alpha,
                                "significant": True,
                            })
                nemenyi_result[
                    "significant_pairs"
                ] = significant_pairs
                nemenyi_result[
                    "unit_of_analysis"
                ] = "seed"
                results["nemenyi"] = nemenyi_result
                # CSV da matriz de p-values
                nemenyi_matrix = pd.DataFrame(
                    p_values,
                    index=models_n,
                    columns=models_n,
                )
                nemenyi_path = (
                    output_dir_path
                    / "nemenyi_p_values.csv"
                )
                nemenyi_matrix.to_csv(
                    nemenyi_path, index=True
                )
                logger.info(
                    "Nemenyi: %d pares "
                    "significativos | "
                    "n=%d seeds | "
                    "matrix salva: %s",
                    len(significant_pairs),
                    len(matrix),
                    nemenyi_path.name,
                )
            except Exception as exc:
                logger.exception(
                    "Nemenyi failed: %s", exc
                )
                results["nemenyi"] = {
                    "error": str(exc),
                    "models": (
                        matrix.columns.tolist()
                    ),
                    "n_observations": int(
                        len(matrix)
                    ),
                    "unit_of_analysis": "seed",
                }
    # ─────────────────────────────────────────
    # Salvar resultados
    # ─────────────────────────────────────────
    output_path = (
        output_dir_path / "statistical_tests.json"
    )
    save_statistical_results(
        results, str(output_path)
    )
    logger.info(
        "Statistical results saved: %s",
        output_path.name,
    )
    # Log sumário
    _log_summary(results, alpha)
    return results
def _log_summary(
    results: Dict[str, Any],
    alpha: float,
) -> None:
    """
    Loga sumário legível dos testes estatísticos.
    """
    logger.info("=" * 60)
    logger.info(
        "STATISTICAL TESTS SUMMARY "
        "(unit=seed, n=%d)",
        results.get("n_observations", "?"),
    )
    kw_res = results.get("kruskal_wallis", {})
    if kw_res:
        sig_datasets = sum(
            1 for v in kw_res.values()
            if isinstance(v, dict)
            and v.get("significant", False)
        )
        total_datasets = sum(
            1 for v in kw_res.values()
            if isinstance(v, dict)
            and not v.get("skipped")
            and not v.get("error")
        )
        logger.info(
            "KW/dataset: %d datasets | "
            "%d significativos (alpha=%.3f)",
            total_datasets, sig_datasets, alpha,
        )
        for dataset, kw in kw_res.items():
            if not isinstance(kw, dict):
                continue
            if kw.get("skipped") or kw.get("error"):
                continue
            if kw.get("significant"):
                sig_pairs = kw.get(
                    "significant_pairs", []
                )
                logger.info(
                    "  [SIG] %s: p=%.4g | "
                    "%d par(es) sig. (Holm)",
                    dataset,
                    kw.get("p_value", 1),
                    len(sig_pairs),
                )
                for pair in sig_pairs:
                    logger.info(
                        "    %s vs %s  p=%.4g",
                        pair.get("model_a", ""),
                        pair.get("model_b", ""),
                        pair.get("p_value", 1),
                    )
    logger.info("-" * 60)
    # Friedman
    friedman = results.get("friedman", {})
    if friedman:
        if friedman.get("skipped"):
            logger.info(
                "Friedman : PULADO (%s)",
                friedman.get("reason", ""),
            )
        elif friedman.get("error"):
            logger.info(
                "Friedman : ERRO (%s)",
                friedman.get("error", ""),
            )
        else:
            logger.info(
                "Friedman : chi2=%.4f  "
                "p=%.4g  sig=%s",
                friedman.get("statistic", 0),
                friedman.get("p_value", 1),
                friedman.get("significant", False),
            )
    # Wilcoxon
    wilcoxon_res = results.get("wilcoxon", [])
    if wilcoxon_res:
        sig = sum(
            1 for r in wilcoxon_res
            if r.get("holm_significant", False)
        )
        logger.info(
            "Wilcoxon  : %d pares | "
            "%d sig. (Holm alpha=%.3f)",
            len(wilcoxon_res), sig, alpha,
        )
        for r in wilcoxon_res:
            if r.get("holm_significant"):
                logger.info(
                    "  [SIG] %s vs %s  "
                    "p=%.4g  holm_alpha=%.4f",
                    r.get("model_a", ""),
                    r.get("model_b", ""),
                    r.get("p_value", 1),
                    r.get("holm_alpha", alpha),
                )
    # Nemenyi
    nemenyi = results.get("nemenyi", {})
    if nemenyi:
        if nemenyi.get("skipped"):
            logger.info(
                "Nemenyi   : PULADO (%s)",
                nemenyi.get("reason", ""),
            )
        elif nemenyi.get("error"):
            logger.info(
                "Nemenyi   : ERRO (%s)",
                nemenyi.get("error", ""),
            )
        else:
            sig_pairs = nemenyi.get(
                "significant_pairs", []
            )
            logger.info(
                "Nemenyi   : %d pares sig.",
                len(sig_pairs),
            )
            for pair in sig_pairs:
                logger.info(
                    "  [SIG] %s vs %s  p=%.4g",
                    pair.get("model_a", ""),
                    pair.get("model_b", ""),
                    pair.get("p_value", 1),
                )
    logger.info("=" * 60)