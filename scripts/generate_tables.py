# -*- coding: utf-8 -*-
"""
scripts/generate_tables.py — Tabelas LaTeX para a monografia.
Versão 6.0 — correções aplicadas:
  - BERT (EN) e BERTimbau (PT) com rótulos distintos.
  - _load_all_raw() agrega por seed corretamente.
  - gen_cross() usa aggfunc="first" apenas após garantir unicidade.
  - gen_intra() implementa destaque do melhor valor por coluna.
  - Legendas especificam "média ± desvio padrão, 10 seeds".
  - _fmt_mean_std() corrigido: $\\pm$ em vez de raw \\\\pm.
  - _to_math() consistente com saída de _fmt_mean_std().
  - _check_seeds_warn() chamada uma vez por linha em gen_intra().
  - Legenda gen_intra() atualizada para 10 seeds.
  - Header de gen_intra() corrigido: \\mu em vez de \\\\mu.
  - Bloco elif morto em best_f1_by_dataset removido.
  - EXPECTED_SEEDS como tupla imutável.
  - pd.concat com sort=False em gen_robustness().
"""
from pathlib import Path
import json
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
RESULTS_DIR    = Path("results/metrics")
AGGREGATED_DIR = Path("results/metrics/aggregated")
OUTPUT_DIR     = Path("latex/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Declarada como tupla para evitar mutação acidental
EXPECTED_SEEDS: Tuple[int, ...] = tuple(range(42, 52))
DATASET_LABELS: Dict[str, str] = {
    "fakebr":             "Fake.br",
    "faketruebr":         "FakeTrue.Br",
    "fakerecogna":        "FakeRecogna",
    "fakenewsnet":        "FakeNewsNet",
    "constraint_covid19": "CONSTRAINT-COVID19",
    "pheme":              "PHEME",
    # aliases legados
    "Fake.br-Corpus":         "Fake.br",
    "FakeTrue.Br":            "FakeTrue.Br",
    "FakeRecogna":            "FakeRecogna",
    "FakeNewsNet-PolitiFact": "FakeNewsNet",
    "CONSTRAINT-COVID19":     "CONSTRAINT-COVID19",
    "PHEME":                  "PHEME",
}
# Rótulos distintos para BERT (EN) e BERTimbau (PT)
MODEL_LABELS: Dict[str, str] = {
    "logistic_regression": r"LR + TF-IDF",
    "mlp":                 r"MLP + TF-IDF",
    "bilstm":              r"BiLSTM",
    "bert":                r"BERT (EN)",
    "bertimbau":           r"BERTimbau (PT)",
}
MODEL_ORDER: List[str] = [
    "logistic_regression", "mlp", "bilstm",
    "bert", "bertimbau",
]
DATASET_ORDER: List[str] = [
    "fakebr", "faketruebr", "fakerecogna",
    "fakenewsnet", "constraint_covid19", "pheme",
]
# ── Helpers ───────────────────────────────────────────────────────────────────
def _fmt_raw(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "---"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)
def _fmt_mean_std(
    mean,
    std,
    decimals: int = 4,
) -> str:
    """
    Formata média ± desvio padrão para LaTeX.
    Produz '$m \\pm s$' diretamente, sem depender de
    _to_math() para inserir o modo matemático.
    """
    if mean is None or (
        isinstance(mean, float) and mean != mean
    ):
        return "---"
    try:
        m = float(mean)
        if std is None or pd.isna(std):
            return f"{m:.{decimals}f}"
        s = float(std)
        return (
            f"${m:.{decimals}f}"
            f" \\pm {s:.{decimals}f}$"
        )
    except Exception:
        return str(mean)
def _to_math(s: str) -> str:
    """
    Envolve em modo matemático se contiver ± (unicode).
    Strings já formatadas por _fmt_mean_std() com $...$
    não precisam passar por esta função.
    """
    if not s or s == "---":
        return s
    if "±" in s:
        return "$" + s.replace("±", r"\pm") + "$"
    return s
def _ds(s: str) -> str:
    return DATASET_LABELS.get(s, s)
def _ml(s: str) -> str:
    return MODEL_LABELS.get(s, s)
def _load(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
def _save(content: str, name: str) -> None:
    out = OUTPUT_DIR / name
    out.write_text(content, encoding="utf-8")
    print(f"[OK] {out}")
def _check_seeds_warn(
    n_found: int,
    label: str = "",
) -> str:
    """
    Retorna asterisco LaTeX se n_found < esperado.
    Imprime aviso UMA vez por chamada — não chamar
    dentro de loops por célula.
    """
    if n_found < len(EXPECTED_SEEDS):
        print(
            f"[AVISO] {label}: apenas {n_found} seed(s) "
            f"(esperadas {len(EXPECTED_SEEDS)})."
        )
        return r"$^{*}$"
    return ""
# ── Carregamento de resultados ────────────────────────────────────────────────
def _load_aggregated() -> pd.DataFrame:
    """
    Tenta carregar resultados pré-agregados de AGGREGATED_DIR.
    Cada arquivo deve conter média/std de todas as métricas.
    Retorna DataFrame vazio se o diretório não existir.
    """
    rows = []
    if not AGGREGATED_DIR.exists():
        return pd.DataFrame()
    for jf in sorted(
        AGGREGATED_DIR.glob("*_aggregated.json")
    ):
        d = _load(jf)
        if not d:
            continue
        metrics_agg = d.get("metrics", {})
        row: Dict[str, Any] = {
            "experiment_prefix": d.get(
                "experiment_prefix", jf.stem
            ),
            "model":   d.get("model", ""),
            "dataset": d.get("dataset", ""),
            "n_runs":  d.get("n_runs", 1),
            "seeds":   d.get("seeds", []),
        }
        for metric, stats in metrics_agg.items():
            if isinstance(stats, dict):
                row[f"{metric}_mean"]      = stats.get("mean")
                row[f"{metric}_std"]       = stats.get("std")
                row[f"{metric}_formatted"] = (
                    d.get("                    metrics_formatted", {}).get(metric)
                )
                for seed, v in zip(
                    d.get("seeds", []),
                    stats.get("values", []),
                ):
                    row[f"{metric}_seed{seed}"] = v
        rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()
def _load_all_raw() -> pd.DataFrame:
    """
    Carrega arquivos EXP_*.json diretamente de RESULTS_DIR,
    agrupando por ID base (sem sufixo _seed{N}) e calculando
    média ± desvio padrão (ddof=1) por grupo.
    Alerta sobre grupos com menos seeds que o esperado.
    Ignora arquivos em subdiretórios (ex: aggregated/).
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for jf in sorted(RESULTS_DIR.glob("EXP_*.json")):
        # Ignora arquivos em subdiretórios
        if jf.parent != RESULTS_DIR:
            continue
        d = _load(jf)
        if not d:
            continue
        base = re.sub(
            r"_seed\d+$",
            "",
            d.get("experiment_id", jf.stem),
        )
        groups[base].append(d)
    metric_keys: Tuple[str, ...] = (
        "f1_macro", "accuracy", "precision_macro",
        "recall_macro", "f1_weighted", "mcc",
        "roc_auc", "pr_auc",
        "f1_macro_ci_lo", "f1_macro_ci_hi",
        "f1_macro_ci_mean",
        "train_time_seconds",
        "inference_time_seconds",
    )
    rows = []
    for base_id, entries in groups.items():
        if not entries:
            continue
        first = entries[0]
        n     = len(entries)
        if n < len(EXPECTED_SEEDS):
            print(
                f"[AVISO] {base_id}: "
                f"apenas {n} seed(s) encontrada(s) "
                f"(esperadas {len(EXPECTED_SEEDS)})."
            )
        row: Dict[str, Any] = {
            "experiment_prefix": base_id,
            "model":   first.get("model", ""),
            "dataset": first.get("dataset", ""),
            "n_runs":  n,
        }
        for metric in metric_keys:
            values: List[float] = []
            for e in entries:
                v = e.get("metrics", {}).get(metric)
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass
            if values:
                mean = float(np.mean(values))
                std  = (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                )
                row[f"{metric}_mean"]      = mean
                row[f"{metric}_std"]       = std
                row[f"{metric}_formatted"] = (
                    f"{mean:.4f} ± {std:.4f}"
                )
                # Valores por seed para rastreabilidade
                for e in entries:
                    v = e.get("metrics", {}).get(metric)
                    seed_str = re.search(
                        r"_seed(\d+)$",
                        e.get("experiment_id", ""),
                    )
                    if seed_str and v is not None:
                        seed_num = seed_str.group(1)
                        row[
                            f"{metric}_seed{seed_num}"
                        ] = float(v)
            else:
                row[f"{metric}_mean"]      = None
                row[f"{metric}_std"]       = None
                row[f"{metric}_formatted"] = None
        rows.append(row)
    return pd.DataFrame(rows)
# ── Tab 1: Intra-dataset (Experimento A) ──────────────────────────────────────
def gen_intra(df: pd.DataFrame) -> str:
    intra = df[
        df["experiment_prefix"].str.startswith("EXP_A")
    ].copy()
    if intra.empty:
        return "% Nenhum resultado EXP_A.\n"
    models_present = [
        m for m in MODEL_ORDER
        if m in intra["model"].values
    ]
    datasets_present = [
        d for d in DATASET_ORDER
        if d in intra["dataset"].values
    ]
    cat_m = pd.Categorical(
        intra["model"], models_present, ordered=True
    )
    cat_d = pd.Categorical(
        intra["dataset"], datasets_present, ordered=True
    )
    intra = intra.assign(
        mc=cat_m, dc=cat_d
    ).sort_values(["mc", "dc"])
    n_seeds_min = int(min(EXPECTED_SEEDS))
    n_seeds_max = int(max(EXPECTED_SEEDS))
    n_seeds_exp = len(EXPECTED_SEEDS)
    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        (
            f"\\caption{{Experimento A --- Intra-dataset. "
            f"Valores reportados como média $\\pm$ desvio padrão "
            f"amostral sobre {n_seeds_exp} execuções independentes "
            f"(seeds {n_seeds_min}--{n_seeds_max}). "
            r"$^{*}$ indica menos de "
            f"{n_seeds_exp} seeds disponíveis. "
            r"\textbf{Negrito} = melhor F1-macro por dataset.}"
        ),
        r"\label{tab:exp_a_intra}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        (
            r"Modelo & Dataset & "
            r"F1-macro ($\mu \pm \sigma$) & "
            r"Acurácia ($\mu \pm \sigma$) & "
            r"Precisão ($\mu \pm \sigma$) & "
            r"Recall ($\mu \pm \sigma$) \\"
        ),
        r"\midrule",
    ]
    # Identifica o melhor F1 por dataset para negrito
    # Bloco elif morto removido — apenas verifica coluna existente
    best_f1_by_dataset: Dict[str, float] = {}
    for ds in datasets_present:
        sub = intra[intra["dataset"] == ds]
        if not sub.empty and "f1_macro_mean" in sub.columns:
            valid = sub["f1_macro_mean"].dropna()
            if not valid.empty:
                best_f1_by_dataset[ds] = float(valid.max())
    prev_m = None
    for _, row in intra.iterrows():
        ml     = _ml(row["model"])
        dl     = _ds(row["dataset"])
        mc     = ml if row["model"] != prev_m else ""
        prev_m = row["model"]
        n_runs = int(row.get("n_runs", len(EXPECTED_SEEDS)))
        # _check_seeds_warn chamada UMA vez por linha —
        # não dentro de _cell() para evitar aviso repetido
        asterisk = _check_seeds_warn(
            n_runs, f"{ml}/{dl}"
        )
        def _cell(
            mean_col: str,
            std_col: str,
        ) -> str:
            mean = row.get(mean_col)
            std  = row.get(std_col)
            return _fmt_mean_std(mean, std)
        f1_str  = _cell(
            "f1_macro_mean", "f1_macro_std"
        ) + asterisk
        acc_str = _cell(
            "accuracy_mean", "accuracy_std"
        )
        pre_str = _cell(
            "precision_macro_mean",
            "precision_macro_std",
        )
        rec_str = _cell(
            "recall_macro_mean", "recall_macro_std"
        )
        # Aplica negrito ao melhor F1 por dataset
        f1_mean_val = row.get("f1_macro_mean")
        best_val    = best_f1_by_dataset.get(
            row["dataset"]
        )
        if (
            f1_mean_val is not None
            and best_val is not None
            and abs(float(f1_mean_val) - best_val) < 1e-8
        ):
            f1_str = r"\textbf{" + f1_str + "}"
        lines.append(
            f"{mc} & {dl} & {f1_str} & {acc_str} "
            f"& {pre_str} & {rec_str} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)
# ── Tab 2: Cross-dataset (Experimento B) ──────────────────────────────────────
def gen_cross() -> str:
    p = RESULTS_DIR / "cross_dataset_summary.csv"
    if not p.exists():
        return (
            "% cross_dataset_summary.csv "
            "não encontrado.\n"
        )
    df        = pd.read_csv(p)
    value_col = (
        "f1_macro_formatted"
        if "f1_macro_formatted" in df.columns
        else "f1_macro"
    )
    tables = []
    models = [
        m for m in MODEL_ORDER
        if m in df["model"].values
    ]
    for model in models:
        sub = df[df["model"] == model].copy()
        sub["trl"] = sub["train_dataset"].map(_ds)
        sub["tel"] = sub["test_dataset"].map(_ds)
        if value_col == "f1_macro_formatted":
            # Deduplicação numérica antes de reconstruir
            # o formatted — garante unicidade no pivot
            mean_col = (
                "f1_macro_mean"
                if "f1_macro_mean" in sub.columns
                else "f1_macro"
            )
            std_col = (
                "f1_macro_std"
                if "f1_macro_std" in sub.columns
                else None
            )
            sub_dedup = (
                sub
                .groupby(
                    ["trl", "tel"], as_index=False
                )[mean_col]
                .mean()
            )
            pivot_data: Dict[
                Tuple[str, str], str
            ] = {}
            for _, r in sub_dedup.iterrows():
                mean_v = r[mean_col]
                std_v: Optional[float] = None
                if std_col:
                    mask = (
                        (sub["trl"] == r["trl"])
                        & (sub["tel"] == r["tel"])
                    )
                    std_rows = (
                        sub[mask][std_col].dropna()
                    )
                    std_v = (
                        float(std_rows.mean())
                        if not std_rows.empty
                        else None
                    )
                pivot_data[
                    (r["trl"], r["tel"])
                ] = _fmt_mean_std(mean_v, std_v)
            all_labels = sorted(
                set(
                    sub["trl"].tolist()
                    + sub["tel"].tolist()
                )
            )
            pivot = pd.DataFrame(
                index=all_labels,
                columns=all_labels,
                dtype=object,
            )
            for (trl, tel), cell in (
                pivot_data.items()
            ):
                pivot.loc[trl, tel] = cell
            pivot = pivot.fillna("---")
        else:
            pivot = (
                sub
                .groupby(["trl", "tel"])["f1_macro"]
                .mean()
                .unstack()
                .round(4)
            )
        caption = (
            f"Exp. B Cross-dataset --- "
            f"{_ml(model)}. "
            r"Linha = treino, Coluna = teste. "
        )
        if value_col == "f1_macro_formatted":
            caption += (
                r"Valores: média $\pm$ $\sigma$ "
                f"({len(EXPECTED_SEEDS)} seeds)."
            )
        else:
            caption += "Valores: F1-macro médio."
        tables.append(
            pivot.to_latex(
                na_rep="---",
                caption=caption,
                label=f"tab:exp_b_{model}",
                escape=False,
            )
        )
    if not tables:
        return (
            "% Nenhum modelo encontrado em "
            "cross_dataset_summary.csv.\n"
        )
    return "\n\n".join(tables)
# ── Tab 3: Combined (Experimento C) ───────────────────────────────────────────
def gen_combined(language: str) -> str:
    p = RESULTS_DIR / "combined_summary.csv"
    if not p.exists():
        return "% combined_summary.csv não encontrado.\n"
    df = pd.read_csv(p).copy()
    if df.empty:
        return (
            f"% Sem resultados para o idioma "
            f"{language}.\n"
        )
    # Filtra por idioma
    if "language" in df.columns:
        lang_col = (
            df["language"]
            .astype(str)
            .str.lower()
            .str.strip()
        )
        if language == "pt":
            df = df[
                lang_col.isin([
                    "pt", "pt-br",
                    "portuguese", "português",
                ])
            ]
        else:
            df = df[
                lang_col.isin([
                    "en", "en-us",
                    "english", "inglês",
                ])
            ]
    elif "test_dataset" in df.columns:
        prefix = f"combined_{language}-"
        df = df[
            df["test_dataset"]
            .astype(str)
            .str.startswith(prefix)
        ]
    if df.empty:
        return (
            f"% Nenhum resultado do Experimento C "
            f"para {language.upper()}.\n"
        )
    # Preserva distinção bert vs bertimbau
    # NÃO remove prefixo de idioma do nome do modelo
    df["ml"] = df["model"].astype(str).map(_ml)
    df["dl"] = (
        df["test_dataset"]
        .astype(str)
        .str.replace(
            r"^combined_(pt|en)-", "", regex=True
        )
        .map(_ds)
    )
    value_col = (
        "f1_macro_formatted"
        if "f1_macro_formatted" in df.columns
        else "f1_macro_mean"
        if "f1_macro_mean" in df.columns
        else "f1_macro"
    )
    # Deduplicação antes do pivot
    df_dedup = (
        df.groupby(["ml        ", "dl"])[value_col]
        .first()
        .reset_index()
    )
    pivot = df_dedup.pivot(
        index="ml", columns="dl", values=value_col
    )
    language_label = (
        "português" if language == "pt" else "inglês"
    )
    pool = (
        "Fake.br + FakeTrue.Br + FakeRecogna"
        if language == "pt"
        else "FakeNewsNet + CONSTRAINT-COVID19 + PHEME"
    )
    return pivot.to_latex(
        na_rep="---",
        caption=(
            f"Exp. C --- Treinamento combinado em "
            f"{language_label} (pool: {pool}). "
            r"Valores: média $\pm$ desvio padrão "
            f"({len(EXPECTED_SEEDS)} seeds)."
        ),
        label=f"tab:exp_c_combined_{language}",
        escape=False,
    )
# ── Tab 4: Temporal (Experimento D) ───────────────────────────────────────────
def gen_temporal() -> str:
    p = RESULTS_DIR / "temporal_summary.csv"
    if not p.exists():
        return "% temporal_summary.csv não encontrado.\n"
    df = pd.read_csv(p).copy()
    df["ml"] = df["model"].map(_ml)
    df["dl"] = df["dataset"].map(_ds)
    value_col = (
        "f1_macro_formatted"
        if "f1_macro_formatted" in df.columns
        else "f1_macro"
    )
    pivot = (
        df.groupby(["ml", "dl"])[value_col]
        .first()
        .unstack()
    )
    caption = r"Exp. D --- Split Temporal. "
    if value_col == "f1_macro_formatted":
        caption += (
            r"Valores: média $\pm$ $\sigma$ "
            f"({len(EXPECTED_SEEDS)} seeds, "
            r"split temporal fixo)."
        )
    else:
        caption += "Valores: F1-macro médio."
    return pivot.to_latex(
        na_rep="---",
        caption=caption,
        label="tab:exp_d_temporal",
        escape=False,
    )
# ── Tab 5: Robustez (Experimento E) ───────────────────────────────────────────
def gen_robustness() -> str:
    files = sorted(RESULTS_DIR.glob("robustness_*.csv"))
    if not files:
        return (
            "% Nenhum robustness_*.csv encontrado.\n"
        )
    df = pd.concat(
        [pd.read_csv(f) for f in files],
        ignore_index=True,
        sort=False,
    )
    df = df[df["perturbation"] != "baseline"].copy()
    df["ml"] = df["model"].map(_ml)
    # Usa delta_f1 (média entre seeds já calculada no CSV)
    avg = (
        df.groupby(
            ["ml", "perturbation"], as_index=False
        )["delta_f1"]
        .mean()
    )
    pivot = avg.pivot(
        index="perturbation",
        columns="ml",
        values="delta_f1",
    ).round(4)
    return pivot.to_latex(
        float_format="%.4f",
        na_rep="---",
        caption=(
            r"Exp. E --- Robustez. "
            r"$\Delta$F1 = F1-baseline $-$ "
            r"F1-perturbado "
            r"(positivo = degradação). "
            r"Médias sobre seeds e datasets."
        ),
        label="tab:exp_e_robustness",
        escape=False,
    )
# ── Tab 6: Generalization Gap ─────────────────────────────────────────────────
def gen_gap() -> str:
    p = RESULTS_DIR / "generalization_gap.csv"
    if not p.exists():
        return (
            "% generalization_gap.csv "
            "não encontrado.\n"
        )
    df = pd.read_csv(p).copy()
    df["Modelo"]  = df["model"].map(_ml)
    df["Dataset"] = df["dataset"].map(_ds)
    intra_col = (
        "f1_intra_formatted"
        if "f1_intra_formatted" in df.columns
        else "f1_intra"
    )
    display_df = df.rename(
        columns={
            intra_col:            "F1 Intra",
            "f1_cross_mean":      "F1 Cross",
            "generalization_gap": "Gap",
        }
    )
    cols = [
        c for c in [
            "Modelo", "Dataset",
            "F1 Intra", "F1 Cross", "Gap",
        ]
        if c in display_df.columns
    ]
    return display_df[cols].to_latex(
        index=False,
        float_format="%.4f",
        na_rep="---",
        caption=(
            r"Generalization Gap: "
            r"F1-macro intra $-$ F1-macro cross médio "
            r"(menor = melhor generalização). "
            r"F1-intra: média $\pm$ $\sigma$ sobre "
            f"{len(EXPECTED_SEEDS)} seeds. "
            r"F1-cross: média não ponderada "
            r"dos pares de teste."
        ),
        label="tab:generalization_gap",
        escape=False,
    )
# ── Tab 7: Rastreabilidade por seed (Experimento A) ───────────────────────────
def gen_per_seed_trace(df: pd.DataFrame) -> str:
    """
    Tabela com valores individuais de F1-macro por seed
    para o Experimento A. Permite verificar consistência
    entre execuções.
    Só é gerada quando existem colunas f1_macro_seed{N}
    no DataFrame, o que ocorre quando os resultados foram
    carregados via _load_aggregated() ou _load_all_raw().
    """
    intra = df[
        df["experiment_prefix"].str.startswith("EXP_A")
    ].copy()
    if intra.empty:
        return (
            "% Nenhum resultado EXP_A "
            "para tabela por seed.\n"
        )
    seed_cols = sorted(
        [
            c for c in intra.columns
            if re.match(r"f1_macro_seed\d+$", c)
        ],
        key=lambda c: int(
            re.search(r"(\d+)$", c).group(1)
        ),
    )
    if not seed_cols:
        return (
            "% Colunas f1_macro_seed* não encontradas. "
            "Verifique se _load_all_raw() extraiu os "
            "valores por seed corretamente.\n"
        )
    models_present = [
        m for m in MODEL_ORDER
        if m in intra["model"].values
    ]
    datasets_present = [
        d for d in DATASET_ORDER
        if d in intra["dataset"].values
    ]
    cat_m = pd.Categorical(
        intra["model"], models_present, ordered=True
    )
    cat_d = pd.Categorical(
        intra["dataset"], datasets_present, ordered=True
    )
    intra = intra.assign(
        mc=cat_m, dc=cat_d
    ).sort_values(["mc", "dc"])
    seed_nums = [        re.search(r"(\d+)$", c).group(1)
        for c in seed_cols
    ]
    seed_header = " & ".join(
        f"Seed {s}" for s in seed_nums
    )
    col_spec = "ll" + "c" * (len(seed_cols) + 2)
    lines = [
        r"\begin{table}[htb]",
        r"\centering",
        (
            r"\caption{Exp. A --- F1-macro por seed "
            r"(rastreabilidade). "
            r"$\mu$ = média amostral, "
            r"$\sigma$ = desvio padrão amostral (ddof=1). "
            r"Seeds: "
            + ", ".join(seed_nums)
            + r".}"
        ),
        r"\label{tab:exp_a_per_seed}",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        (
            rf"Modelo & Dataset & {seed_header}"
            rf" & $\mu$ & $\sigma$ \\"
        ),
        r"\midrule",
    ]
    prev_m = None
    for _, row in intra.iterrows():
        ml     = _ml(row["model"])
        dl     = _ds(row["dataset"])
        mc     = ml if row["model"] != prev_m else ""
        prev_m = row["model"]
        seed_vals = " & ".join(
            (
                f"{float(row[c]):.4f}"
                if (
                    row.get(c) is not None
                    and not (
                        isinstance(row[c], float)
                        and row[c] != row[c]
                    )
                )
                else "---"
            )
            for c in seed_cols
        )
        mean_v = row.get("f1_macro_mean")
        std_v  = row.get("f1_macro_std")
        mean_s = (
            f"{float(mean_v):.4f}"
            if mean_v is not None
            else "---"
        )
        std_s = (
            f"{float(std_v):.4f}"
            if std_v is not None
            else "---"
        )
        lines.append(
            f"{mc} & {dl} & {seed_vals} "
            f"& {mean_s} & {std_s} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)
# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "=" * 62)
    print("  generate_tables — multi-seed (v6)")
    print(f"  Saída: {OUTPUT_DIR.resolve()}")
    print(f"  Seeds esperadas: {list(EXPECTED_SEEDS)}")
    print("=" * 62 + "\n")
    # Tenta carregar resultados pré-agregados
    df = _load_aggregated()
    if df.empty:
        print(
            "  [INFO] Resultados pré-agregados não "
            f"encontrados em {AGGREGATED_DIR}.\n"
            "         Carregando diretamente dos "
            "JSONs por seed..."
        )
        df = _load_all_raw()
    if df.empty:
        print(
            "[AVISO] Nenhum resultado encontrado em "
            f"{RESULTS_DIR}. "
            "Verifique se os experimentos foram "
            "executados."
        )
        sys.exit(0)
    print(
        f"  Carregados {len(df)} grupos de "
        f"experimentos "
        f"(após agregação por seed).\n"
    )
    _save(gen_intra(df),          "intra_dataset.tex")
    _save(
        gen_per_seed_trace(df),
        "intra_dataset_per_seed.tex",
    )
    _save(gen_cross(),            "cross_dataset.tex")
    _save(gen_combined("pt"),     "combined_pt.tex")
    _save(gen_combined("en"),     "combined_en.tex")
    _save(gen_temporal(),         "temporal.tex")
    _save(gen_robustness(),       "robustness.tex")
    _save(gen_gap(),              "generalization_gap.tex")
    print(
        f"\n[PRONTO] Tabelas LaTeX em "
        f"{OUTPUT_DIR.resolve()}/"
    )
    print(
        "  Uso no Overleaf: "
        r"\input{tables/intra_dataset}"
    )
    print(
        "  Uso no Overleaf: "
        r"\input{tables/intra_dataset_per_seed}"
    )
if __name__ == "__main__":
    main()