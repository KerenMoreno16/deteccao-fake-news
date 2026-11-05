"""
generate_report.py — Gerador automático de fragmentos LaTeX.
Correções aplicadas:
  - load_all_results() agrega por seed (média ± std) em vez de
    retornar uma linha por seed individual.
  - gen_exp_A_intra() usa f1_macro (média) e f1_macro_std.
  - gen_exp_B_cross() remove sufixo _seed{N} antes do parse do ID.
  - gen_best_model_paragraph() e gen_full_summary() usam médias.
  - Legenda das tabelas especifica "média ± desvio padrão, 10 seeds".
  - BERT (EN) e BERTimbau (PT) distinguidos nos rótulos.
  - _escape() corrigido: escapes simples em vez de duplos.
  - _pct() corrigido: \\% em vez de \\\\%.
  - \\\\begin corrigido para \\begin em todas as tabelas.
  - metric_keys sem duplicatas.
  - seed_note com ramos distintos.
  - Entidades HTML &#x74; substituídas por texto literal.
  - f-strings usadas onde interpolação é necessária.
Fragmentos gerados em latex/report_fragments/:
  datasets_summary.tex | exp_A_intra.tex | exp_A_full_metrics.tex
  exp_B_cross.tex | exp_C_combined.tex | exp_D_temporal.tex
  exp_E_robustness.tex | generalization_gap.tex
  comparison_best_model.tex | train_time.tex | full_report_summary.tex
  master_report.tex
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR    = ROOT / "results" / "metrics"
OUTPUT_DIR     = ROOT / "latex" / "report_fragments"
EXPECTED_SEEDS: Tuple[int, ...] = tuple(range(42, 52))  # 10 seeds: 42..51
DATASET_LABELS: Dict[str, str] = {
    "faketruebr":         "FakeTrue.Br",
    "fakebr":             "Fake.br",
    "fakerecogna":        "FakeRecogna",
    "fakenewsnet":        "FakeNewsNet",
    "constraint_covid19": "CONSTRAINT-COVID19",
    "pheme":              "PHEME",
}
MODEL_LABELS: Dict[str, str] = {
    "logistic_regression": r"LR + TF-IDF",
    "mlp":                 r"MLP + TF-IDF",
    "bilstm":              r"BiLSTM",
    "bert":                r"BERT (EN)",
    "bertimbau":           r"BERTimbau (PT)",
}
METRIC_LABELS: Dict[str, str] = {
    "accuracy":        "Acurácia",
    "precision_macro": "Precisão (macro)",
    "recall_macro":    "Recall (macro)",
    "f1_macro":        "F1-macro",
    "f1_weighted":     "F1-weighted",
    "roc_auc":         "ROC-AUC",
    "pr_auc":          "PR-AUC",
    "mcc":             "MCC",
}
# ── Helpers ───────────────────────────────────────────────────────────────────
def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "---"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)
def _fmt_mean_std(mean, std) -> str:
    if mean is None or (
        isinstance(mean, float) and mean != mean
    ):
        return "---"
    try:
        m = float(mean)
        s = float(std) if (
            std is not None
            and not (isinstance(std, float) and std != std)
        ) else 0.0
        if s > 0:
            return f"{m:.4f} $\\pm$ {s:.4f}"
        return f"{m:.4f}"
    except (TypeError, ValueError):
        return str(mean)
def _pct(v) -> str:
    if v is None:
        return "---"
    try:
        return f"{float(v)*100:.2f}\\%"
    except (TypeError, ValueError):
        return str(v)
def _escape(s: str) -> str:
    """
    Escapa caracteres especiais LaTeX.
    Cada sequência produz exatamente uma barra invertida no LaTeX.
    """
    replacements = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\^{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s
def _ds_label(name: str) -> str:
    return DATASET_LABELS.get(name, _escape(name))
def _model_label(name: str) -> str:
    return MODEL_LABELS.get(name, _escape(name))
def _header_comment(title: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"% {'='*68}\n"
        f"% {title}\n"
        f"% Gerado automaticamente por generate_report.py em {ts}\n"
        f"% NÃO EDITE MANUALMENTE — re-gere rodando:\n"
        f"%   python scripts/generate_report.py\n"
        f"% {'='*68}\n\n"
    )
def _load_json(path: Path) -> Dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
# ── Carregamento agregado por seed ────────────────────────────────────────────
def load_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Carrega todos os JSONs EXP_*.json agrupando por seed.
    Cada linha do DataFrame representa a MÉDIA ± STD das métricas
    sobre as seeds disponíveis para aquele experimento base.
    Alerta quando um experimento tem menos seeds que o esperado.
    """
    groups: Dict[str, List[Dict]] = {}
    for p in sorted(results_dir.glob("EXP_*.json")):
        data = _load_json(p)
        if not data:
            continue
        base_id = re.sub(
            r"_seed\d+$",
            "",
            data.get("experiment_id", p.stem),
        )
        if base_id not in groups:
            groups[base_id] = []
        groups[base_id].append(data)
    # Sem duplicatas — cada métrica processada uma única vez
    metric_keys: Tuple[str, ...] = (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
        "pr_auc",
        "mcc",
        "f1_macro_ci_lo",
        "f1_macro        _ci_hi",
        "train_time_s",
        "train_time_seconds",
        "inference_time_seconds",
    )
    rows = []
    for base_id, entries in groups.items():
        if not entries:
            continue
        first = entries[0]
        row: Dict[str, Any] = {
            "experiment_id": base_id,
            "model":         first.get("model", ""),
            "dataset":       first.get("dataset", ""),
            "train_dataset": first.get("train_dataset", ""),
            "test_dataset":  first.get("test_dataset", ""),
            "split_type":    first.get("split_type", "random"),
            "train_size":    first.get("train_size"),
            "test_size":     first.get("test_size"),
            "n_seeds":       len(entries),
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
                row[metric]              = float(np.mean(values))
                row[f"{metric}_std"]     = (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                )
                row[f"{metric}_n"]       = len(values)
            else:
                row[metric]              = None
                row[f"{metric}_std"]     = None
                row[f"{metric}_n"]       = 0
        # Retrocompatibilidade: train_time_s -> train_time_seconds
        if row.get("train_time_s") is None:
            row["train_time_s"] = row.get("train_time_seconds")
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        low = df[df["n_seeds"] < len(EXPECTED_SEEDS)]
        for _, r in low.iterrows():
            print(
                f"[AVISO] {r['experiment_id']}: "
                f"apenas {r['n_seeds']} seed(s) encontrada(s) "
                f"(esperadas {len(EXPECTED_SEEDS)})."
            )
    return df
# ── Fragmento 1 — Resumo dos datasets ────────────────────────────────────────
def gen_datasets_summary(results_dir: Path) -> str:
    content = _header_comment("Tabela: Resumo dos datasets")
    meta = {
        "FakeTrue.Br": {
            "idioma":  "PT-BR",
            "domínio": "Política",
            "fonte":   r"Chavarro et al.~\cite{chavarro2023faketruebr}",
        },
        "Fake.br": {
            "idioma":  "PT-BR",
            "domínio": "Geral",
            "fonte":   r"Monteiro et al.~\cite{monteiro2018fakebr}",
        },
        "FakeRecogna": {
            "idioma":  "PT-BR",
            "domínio": "Política",
            "fonte":   r"Garcia et al.~\cite{garcia2023fakerecogna}",
        },
        "FakeNewsNet (PolitiFact)": {
            "idioma":  "EN",
            "domínio": "Política",
            "fonte":   r"Shu et al.~\cite{shu2018fakenewsnet}",
        },
        "CONSTRAINT-COVID19": {
            "idioma":  "EN",
            "domínio": "Saúde",
            "fonte":   r"Patwa et al.~\cite{patwa2021constraint}",
        },
        "PHEME": {
            "idioma":  "EN",
            "domínio": "Múltiplo",
            "fonte":   r"Zubiaga et al.~\cite{zubiaga2016pheme}",
        },
    }
    content += (
        r"\begin{table}[htbp]" + "\n"
        r"  \centering" + "\n"
        r"  \caption{Datasets utilizados neste trabalho.}" + "\n"
        r"  \label{tab:datasets_summary}" + "\n"
        r"  \begin{tabular}{lllp{5cm}}" + "\n"
        r"    \hline" + "\n"
        r"    \textbf{Dataset} & \textbf{Idioma} & "
        r"\textbf{Domínio} & \textbf{Referência} \\" + "\n"
        r"    \hline" + "\n"
    )
    for ds, info in meta.items():
        content += (
            f"    {_escape(ds)} & {info['idioma']} & "
            f"{info['domínio']} & {info['fonte']} \\\\\n"
        )
    content += (
        r"    \hline" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    return content
# ── Fragmento 2 — Experimento A: Intra-dataset ───────────────────────────────
def gen_exp_A_intra(df: pd.DataFrame) -> str:
    """
    Tabela LaTeX do Experimento A usando médias agregadas por seed.
    Células mostram 'média ± desvio padrão'.
    Asterisco (*) indica menos de 10 seeds disponíveis.
    """
    content = _header_comment(
        "Tabela: Experimento A — Resultados Intra-dataset"
    )
    intra = df[
        df["experiment_id"].str.startswith("EXP_A")
    ].copy()
    if intra.empty:
        return (
            content
            + "% Nenhum resultado EXP_A encontrado ainda.\n"
        )
    models   = [
        "logistic_regression", "mlp",
        "bilstm", "bertimbau",
    ]
    datasets = sorted(
        intra["dataset"].dropna().unique().tolist()
    )
    col_spec  = "l" + "c" * len(datasets)
    ds_header = " & ".join(
        [
            r"\textbf{" + _ds_label(d) + "}"
            for d in datasets
        ]
    )
    n_seeds_expected = len(EXPECTED_SEEDS)
    content += (
        r"\begin{table}[htbp]" + "\n"
        r"  \centering" + "\n"
        f"  \\caption{{Resultados do Experimento A --- F1-macro "
        f"($\\mu \\pm \\sigma$) por modelo e dataset. "
        f"Cada célula reporta a média e o desvio padrão amostral "
        f"sobre {n_seeds_expected} execuções independentes. "
        f"$^{{*}}$ indica menos de {n_seeds_expected} "
        f"seeds disponíveis.}}\n"
        r"  \label{tab:exp_a_intra}" + "\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        r"    \hline" + "\n"
        f"    \\textbf{{Modelo}} & {ds_header} \\\\\n"
        r"    \hline" + "\n"
    )
    for model in models:
        sub      = intra[intra["model"] == model]
        row_vals = []
        for ds in datasets:
            m_row = sub[sub["dataset"] == ds]
            if m_row.empty:
                row_vals.append("---")
                continue
            f1_mean = m_row["f1_macro"].values[0]
            f1_std  = (
                m_row["f1_macro_std"].values[0]
                if "f1_macro_std" in m_row.columns
                else None
            )
            n_seeds = (
                int(m_row["n_seeds"].values[0])
                if "n_seeds" in m_row.columns
                else len(EXPECTED_SEEDS)
            )
            if f1_mean is None:
                row_vals.append("---")
                continue
            cell = _fmt_mean_std(f1_mean, f1_std)
            if n_seeds < len(EXPECTED_SEEDS):
                cell += r"$^{*}$"
            row_vals.append(cell)
        content += (
            f"    {_model_label(model)} & "
            + " & ".join(row_vals)
            + " \\\\\n"
        )
    content += (
        r"    \hline" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}" + "\n\n"
    )
    # Análise automática baseada em médias
    if "f1_macro" in intra.columns:
        best_rows = (
            intra
            .loc[intra.groupby("dataset")["f1_macro"].idxmax()]
            .copy()
        )
        paragraphs = []
        for _, row in best_rows.iterrows():
            f1_mean = row["f1_macro"]
            f1_std  = row.get("f1_macro_std")
            val_str = _fmt_mean_std(f1_mean, f1_std)
            paragraphs.append(
                f"Para o dataset "
                f"\\textit{{{_ds_label(row['dataset'])}}}, "
                f"o melhor modelo em média foi "
                f"\\textbf{{{_model_label(row['model'])}}} "
                f"com F1-macro = {val_str}."
            )
        if paragraphs:
            content += (
                "\n% --- Análise automática "
                "(médias sobre seeds) ---\n"
                r"\paragraph{Análise}" + "\n"
                + " ".join(paragraphs) + "\n\n"
            )
    return content
# ── Fragmento 3 — Experimento A: Métricas completas ──────────────────────────
def gen_exp_A_full_metrics(df: pd.DataFrame) -> str:
    """
    Tabela detalhada por dataset com todas as métricas (médias).
    """
    content = _header_comment(
        "Tabela: Experimento A — Métricas Completas"
    )
    intra = df[
        df["experiment_id"].str.startswith("EXP_A")
    ].copy()
    if intra.empty:
        return (
            content
            + "% Nenhum resultado EXP_A encontrado.\n"
        )
    metrics = [
        "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "roc_auc", "mcc",
    ]
    metric_headers = " & ".join(
        [
            r"\textbf{" + METRIC_LABELS.get(m, m) + "}"
            for m in metrics
        ]
    )
    datasets = sorted(
        intra["dataset"].dropna().unique().tolist()
    )
    for ds in datasets:
        sub      = intra[intra["dataset"] == ds]
        col_spec = "l" + "c" * len(metrics)
        content += (
            f"\n% Dataset: {_ds_label(ds)}\n"
            r"\begin{table}[htbp]" + "\n"
            r"  \centering" + "\n"
            f"  \\caption{{Métricas do Experimento A --- "
            f"\\textit{{{_ds_label(ds)}}} "
            f"(médias sobre {len(EXPECTED_SEEDS)} seeds).}}\n"
            f"  \\label{{tab:exp_a_{ds}_metrics}}\n"
            f"  \\begin{{tabular}}{{{col_spec}}}\n"
            r"    \hline" + "\n"
            f"    \\textbf{{Modelo}} & {metric_headers} \\\\\n"
            r"    \hline" + "\n"
        )
        for model in [
            "logistic_regression", "mlp",
            "bilstm", "bertimbau",
        ]:
            m_row = sub[sub["model"] == model]
            if m_row.empty:
                content += (
                    f"    {_model_label(model)} & "
                    + " & ".join(["---"] * len(metrics))
                    + " \\\\\n"
                )
            else:
                vals = [
                    _fmt(m_row[m].values[0])
                    for m in metrics
                ]
                content += (
                    f"    {_model_label(model)} & "
                    + " & ".join(vals)
                    + " \\\\\n"
                )
        content += (
            r"    \hline" + "\n"
            r"  \end{tabular}" + "\n"
            r"\end{table}" + "\n"
        )
    return content
# ── Fragmento 4 — Experimento B: Cross-dataset ───────────────────────────────
def gen_exp_B_cross(df: pd.DataFrame) -> str:
    """
    Tabelas cross-dataset com parse robusto do experiment_id.
    Remove sufixo _seed{N} antes de extrair train/test.
    Células mostram médias agregadas sobre seeds.
    """
    content = _header_comment(
        "Tabelas: Experimento B — Cross-dataset"
    )
    cross = df[
        df["experiment_id"].str.startswith("EXP_B")
    ].copy()
    if cross.empty:
        return (
            content
            + "% Nenhum resultado EXP_B encontrado.\n"
        )
    def _parse_cross_id(
        exp_id: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extrai (train_ds, test_ds) do experiment_id.
        Remove sufixo _seed{N} antes do parse.
        """
        clean_id = re.sub(r"_seed\d+$", "", exp_id)
        try:
            after_model = clean_id.split("_TRAIN_", 1)[1]
            train_ds, test_ds = after_model.split(
                "_TEST_", 1
            )
            return train_ds.strip(), test_ds.strip()
        except (IndexError, ValueError):
            return None, None
    cross["train_ds"] = cross["experiment_id"].apply(
        lambda x: _parse_cross_id(x)[0]
    )
    cross["test_ds"] = cross["experiment_id"].apply(
        lambda x: _parse_cross_id(x)[1]
    )
    n_before = len(cross)
    cross    = cross.dropna(subset=["train_ds", "test_ds"])
    n_after  = len(cross)
    if n_after < n_before:
        print(
            f"[AVISO] gen_exp_B_cross: "
            f"{n_before - n_after} linha(s) descartadas "
            f"por falha no parse do experiment_id."
        )
    if cross.empty:
        return (
            content
            + "% Nenhum resultado EXP_B com parse válido.\n"
        )
    models = [
        "logistic_regression", "mlp",
        "bilstm", "bertimbau",
    ]
    for model in models:
        sub = cross[cross["model"] == model].copy()
        if sub.empty:
            continue
        all_ds = sorted(
            set(
                sub["train_ds"].tolist()
                + sub["test_ds"].tolist()
            )
        )
        col_spec  = "l" + "c" * len(all_ds)
        ds_header = " & ".join(
            [
                r"\textbf{" + _ds_label(d) + "}"
                for d in all_ds
            ]
        )
        content += (
            f"\n% Modelo: {_model_label(model)}\n"
            r"\begin{table}[htbp]" + "\n"
            r"  \centering" + "\n"
            f"  \\caption{{Experimento B --- "
            f"Generalização cross-dataset para "
            f"\\textbf{{{_model_label(model)}}} "
            f"(F1-macro, médias sobre "
            f"{len(EXPECTED_SEEDS)} seeds). "
            f"Linhas = treino, Colunas = teste.}}\n"
            f"  \\label{{tab:cross_{model}}}\n"
            f"  \\begin{{tabular}}{{{col_spec}}}\n"
            r"    \hline" + "\n"
            f"    \\textbf{{Treino $\\downarrow$ / "
            f"Teste $\\rightarrow$}} & "
            f"{ds_header} \\\\\n"
            r"    \hline" + "\n"
        )
        for train in all_ds:
            row_vals = []
            for test in all_ds:
                if train == test:
                    row_vals.append("---")
                    continue
                match = sub[
                    (sub["train_ds"] == train)
                    & (sub["test_ds"] == test)
                ]
                if match.empty:
                    row_vals.append("---")
                    continue
                f1_mean = match["f1_macro"].values[0]
                f1_std  = (
                    match["f1_macro_std"].values[0]
                    if "f1_macro_std" in match.columns
                    else None
                )
                row_vals.append(
                    _fmt_mean_std(f1_mean, f1_std)
                )
            content += (
                f"    {_ds_label(train)} & "
                + " & ".join(row_vals)
                + " \\\\\n"
            )
        content += (
            r"    \hline" + "\n"
            r"  \end{tabular}" + "\n"
            r"\end{table}" + "\n"
        )
    # Análise textual baseada em médias
    if not cross.empty:
        model_means = (
            cross.groupby("model")["f1_macro"]
            .mean()
            .sort_values(ascending=False)
        )
        if not model_means.empty:
            best_gen  = model_means.index[0]
            best_val  = model_means.iloc[0]
            worst_gen = model_means.index[-1]
            worst_val = model_means.iloc[-1]
            content += (
                "\n% --- Análise automática ---\n"
                r"\paragraph{Análise de generalização}" + "\n"
                f"O modelo com maior F1-macro médio no cenário "
                f"cross-dataset foi "
                f"\\textbf{{{_model_label(best_gen)}}} "
                f"(média = {_fmt(best_val)}). "
                f"O modelo com menor média foi "
                f"\\textbf{{{_model_label(worst_gen)}}} "
                f"(média = {_fmt(worst_val)}). "
                f"Valores representam médias sobre pares "
                f"treino/teste e sobre as "
                f"{len(EXPECTED_SEEDS)} seeds.\n\n"
            )
    return content
# ── Fragmento 5 — Experimento C: Combined ────────────────────────────────────
def gen_exp_C_combined(df: pd.DataFrame) -> str:
    content = _header_comment(
        "Tabela: Experimento C — Combined PT-BR"
    )
    combined = df[
        df["experiment_id"].str.startswith("EXP_C")
    ].copy()
    if combined.empty:
        return (
            content
            + "% Nenhum resultado EXP_C encontrado.\n"
        )
    models     = [
        "logistic_regression", "mlp",
        "bilstm", "bertimbau",
    ]
    datasets_c = sorted(
        combined["dataset"].dropna().unique().tolist()
    )
    col_spec  = "l" + "c" * len(datasets_c)
    ds_header = " & ".join(
        [
            r"\textbf{" + _ds_label(d) + "}"
            for d in datasets_c
        ]
    )
    content += (
        r"\begin{table}[htbp]" + "\n"
        r"  \centering" + "\n"
        f"  \\caption{{Experimento C --- Treinamento no corpus "
        f"PT combinado; avaliação por dataset "
        f"(F1-macro, médias sobre "
        f"{len(EXPECTED_SEEDS)} seeds).}}\n"
        r"  \label{tab:exp_c_combined}" + "\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        r"    \hline" + "\n"
        f"    \\textbf{{Modelo}} & {ds_header} \\\\\n"
        r"    \hline" + "\n"
    )
    for model in models:
        sub  = combined[combined["model"] == model]
        vals = []
        for d in datasets_c:
            m_row = sub[sub["dataset"] == d]
            if m_row.empty:
                vals.append("---")
            else:
                f1_mean = m_row["f1_macro"].values[0]
                f1_std  = (
                    m_row["f1_macro_std"].values[0]
                    if "f1_macro_std" in m_row.columns
                    else None
                )
                vals.append(
                    _fmt_mean_std(f1_mean, f1_std)
                )
        content += (
            f"    {_model_label(model)} & "
            + " & ".join(vals)
            + " \\\\\n"
        )
    content += (
        r"    \hline" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    return content
# ── Fragmento 6 — Experimento D: Temporal ────────────────────────────────────
def gen_exp_D_temporal(df: pd.DataFrame) -> str:
    content = _header_comment(
        "Tabela: Experimento D — Análise Temporal"
    )
    temporal = df[
        df["experiment_id"].str.startswith("EXP_D")
    ].copy()
    if temporal.empty:
        return (
            content
            + "% Nenhum resultado EXP_D encontrado "
            + "(requer timestamps nos dados).\n"
        )
    # Carrega JS divergence dos JSONs (pode estar ausente)
    js_data: Dict[str, List[float]] = {}
    for p in sorted(RESULTS_DIR.glob("EXP_D_*.json")):
        d      = _load_json(p)
        exp_id = re.sub(
            r"_seed\d+$", "",
            d.get("experiment_id", p.stem),
        )
        # js_divergence_unigrams gravado na raiz do JSON
        # via extra_meta; fallback para "metrics" por
        # retrocompatibilidade.
        jsd = d.get("js_divergence_unigrams") or (
            d.get("metrics", {})
            .get("js_divergence_unigrams")
        )
        if jsd is not None:
            if exp_id not in js_data:
                js_data[exp_id] = []
            js_data[exp_id].append(float(jsd))
    # Média da JSD entre seeds
    js_means: Dict[str, float] = {
        exp_id: float(np.mean(vals))
        for exp_id, vals in js_data.items()
    }
    jsd_available = bool(js_means)
    if not jsd_available:
        print(
            "[AVISO] gen_exp_D_temporal: "
            "js_divergence_unigrams ausente em todos "
            "os JSONs. Integre calcular_jsd_classes "
            "ao pipeline EXP_D."
        )
    jsd_note = (
        r" \textit{JSD não disponível: "
        r"integrar calcular\_jsd\_classes ao pipeline.}"
        if not jsd_available
        else ""
    )
    content += (
        r"\begin{table}[htbp]" + "\n"
        r"  \centering" + "\n"
        f"  \\caption{{Experimento D --- Análise temporal: "
        f"F1-macro (médias sobre {len(EXPECTED_SEEDS)} seeds) "
        f"e divergência Jensen-Shannon treino/teste "
        f"(bits, base 2).{jsd_note}}}\n"
        r"  \label{tab:exp_d_temporal}" + "\n"
        r"  \begin{tabular}{llcc}" + "\n"
        r"    \hline" + "\n"
        r"    \textbf{Modelo} & \textbf{Dataset} & "
        r"\textbf{F1-macro} & "
        r"\textbf{JS-div. (bits)} \\" + "\n"
        r"    \hline" + "\n"
    )
    for _, row in temporal.iterrows():
        exp_id  = row["experiment_id"]
        jsd     = js_means.get(exp_id)
        js_str  = _fmt(jsd) if jsd is not None else "---"
        f1_mean = row["f1_macro"]
        f1_std  = (
            row["f1_macro_std"]
            if "f1_macro_std" in row.index
            else None
        )
        content += (
            f"    {_model_label(row['model'])} & "
            f"{_ds_label(row['dataset'])} & "
            f"{_fmt_mean_std(f1_mean, f1_std)} & "
            f"{js_str} \\\\\n"
        )
    content += (
        r"    \hline" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    return content
# ── Fragmento 7 — Experimento E: Robustez ────────────────────────────────────
def gen_exp_E_robustness(results_dir: Path) -> str:
    content = _header_comment(
        "Tabela: Experimento E — Robustez"
    )
    rob_files = sorted(results_dir.glob("EXP_E_*.json"))
    if not rob_files:
        return (
            content
            + "% Nenhum resultado EXP_E encontrado.\n"
        )
    # Agrupa por (model, dataset, perturbation) sobre seeds
    groups: Dict[Tuple[str, str, str], List[float]] = {}
    for p in rob_files:
        d       = _load_json(p)
        model   = d.get("model", "")
        dataset = d.get("dataset", "")
        for pert_name, pert_data in (
            d.get("metrics", {}).items()
        ):
            if not isinstance(pert_data, dict):
                continue
            delta = pert_data.get("delta_f1")
            if delta is None:
                continue
            key = (model, dataset, pert_name)
            if key not in groups:
                groups[key] = []
            try:
                groups[key].append(float(delta))
            except (TypeError, ValueError):
                pass
    if not groups:
        return (
            content
            + "% Dados de robustez sem estrutura "
            + "reconhecida.\n"
        )
    # Agrega por (model, perturbation) —
    # média sobre datasets e seeds
    agg: Dict[Tuple[str, str], List[float]] = {}
    for (model, _dataset, pert), deltas in groups.items():
        key = (model, pert)
        if key not in agg:
            agg[key] = []
        agg[key].extend(deltas)
    models = [
        "logistic_regression", "mlp",
        "bilstm", "bertimbau",
    ]
    perturbations = sorted(
        set(k[1] for k in agg.keys())
    )
    pert_labels: Dict[str, str] = {
        "remove_urls":        "Rem. URLs",
        "remove_hashtags":    "Rem. Hashtags",
        "remove_mentions":    "Rem. Menções",
        "remove_emojis":      "Rem. Emojis",
        "remove_all_special": "Rem. Especiais",
        "alphanumeric_only":  "Somente Alfanum.",
        "truncate_20_tokens": "Trunca 20 tokens",
    }
    col_spec     = "l" + "c" * len(models)
    model_header = " & ".join(
        [
            r"\textbf{" + _model_label(m) + "}"
            for m in models
        ]
    )
    content += (
        r"\begin{table}[htbp]" + "\n"
        r"  \centering" + "\n"
        r"  \caption{Experimento E --- "
        r"$\Delta$F1-macro médio "
        r"sob perturbações linguísticas "
        r"(positivo = degradação de desempenho). "
        r"Médias sobre seeds e datasets.}" + "\n"
        r"  \label{tab:exp_e_robustness}" + "\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        r"    \hline" + "\n"
        f"    \\textbf{{Perturbação}} & "
        f"{model_header} \\\\\n"
        r"    \hline" + "\n"
    )
    for pert in perturbations:
        row_vals = []
        for model in models:
            vals = agg.get((model, pert), [])
            if not vals:
                row_vals.append("---")
            else:
                row_vals.append(_fmt(float(np.mean(vals))))
        label = pert_labels.get(pert, _escape(pert))
        content += (
            f"    {label} & "
            + " & ".join(row_vals)
            + " \\\\\n"
        )
    content += (
        r"    \hline" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    return content
# ── Fragmento 8 — Generalization Gap ─────────────────────────────────────────
def gen_generalization_gap(results_dir: Path) -> str:
    content  = _header_comment("Tabela: Generalization Gap")
    gap_path = results_dir / "generalization_gap.csv"
    if not gap_path.exists():
        return (
            content
            + "% generalization_gap.csv não encontrado. "
            + "Execute build_summary_csvs.py primeiro.\n"
        )
    gap_df  = pd.read_csv(gap_path)
    models  = [
        "logistic_regression", "mlp",
        "bilstm", "bertimbau",
    ]
    datasets = (
        sorted(
            gap_df["dataset"].dropna().unique().tolist()
        )
        if "dataset" in gap_df.columns
        else []
    )
    content += (
        r"\begin{table}[htbp]" + "\n"
        r"  \centering" + "\n"
        r"  \caption{Generalization Gap: F1-macro "
        r"intra-dataset $-$ F1-macro médio cross-dataset "
        r"(menor valor = melhor generalização). "
        r"F1-intra reportado como média $\pm$ desvio "
        r"padrão sobre 10 seeds. "
        r"Nota: F1-cross é a média não ponderada "
        r"dos pares de teste disponíveis.}" + "\n"
        r"  \label{tab:gen_gap}" + "\n"
    )
    if datasets:
        col_spec  = "l" + "c" * len(datasets)
        ds_header = " & ".join(
            [
                r"\textbf{" + _ds_label(d) + "}"
                for d in datasets
            ]
        )
        content += (
            f"  \\begin{{tabular}}{{{col_spec}}}\n"
            r"    \hline" + "\n"
            f"    \\textbf{{Modelo}} & "
            f"{ds_header} \\\\\n"
            r"    \hline" + "\n"
        )
        for model in models:
            sub  = gap_df[gap_df["model"] == model]
            vals = []
            for ds in datasets:
                m_sub = (
                    sub[sub["dataset"] == ds]
                    if "dataset" in sub.columns
                    else pd.DataFrame()
                )
                if m_sub.empty:
                    vals.append("---")
                else:
                    vals.append(
                        _fmt(
                            m_sub[
                                "generalization_gap"
                            ].values[0]
                        )
                    )
            content += (
                f"    {_model_label(model)} & "
                + " & ".join(vals)
                + " \\\\\n"
            )
    else:
        content += (
            r"  \begin{tabular}{lcc}" + "\n"
            r"    \hline" + "\n"
            r"    \textbf{Modelo} & "
            r"\textbf{F1 Intra ($\mu \pm \sigma$)} & "
            r"\textbf{Gap médio} \\" + "\n"
            r"    \hline" + "\n"
        )
        for model in models:
            sub = gap_df[gap_df["model"] == model]
            if sub.empty:
                content += (
                    f"    {_model_label(model)} "
                    f"& --- & --- \\\\\n"
                )
                continue
            avg_gap   = sub["generalization_gap"].mean()
            intra_col = (
                "f1_intra_formatted"
                if "f1_intra_formatted" in sub.columns
                else "f1_intra"
            )
            intra_val = sub[intra_col].values[0]
            content += (
                f"    {_model_label(model)} & "
                f"{intra_val} & "
                f"{_fmt(avg_gap)} \\\\\n"
            )
    content += (
        r"    \hline" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    return content
# ── Fragmento 9 — Melhor modelo por dataset ───────────────────────────────────
def gen_best_model_paragraph(df: pd.DataFrame) -> str:
    """
    Identifica o melhor modelo por dataset usando a MÉDIA sobre
    seeds, não um valor de seed individual.
    """
    content = _header_comment(
        "Parágrafo: Melhor modelo por dataset"
    )
    intra = df[
        df["experiment_id"].str.startswith("EXP_A")
    ].copy()
    if intra.empty:
        return (
            content
            + "% Nenhum resultado disponível.\n"
        )
    if "f1_macro" not in intra.columns:
        return (
            content
            + "% Coluna f1_macro ausente. "
            + "Execute build_summary_csvs.py.\n"
        )
    lines = [
        r"\paragraph{Melhor modelo por dataset}",
        "",
        r"A \autoref{tab:exp_a_intra} apresenta os "
        r"resultados do Experimento A. "
        f"Os valores correspondem à média sobre "
        f"{len(EXPECTED_SEEDS)} execuções independentes "
        f"(seeds "
        f"{min(EXPECTED_SEEDS)}--{max(EXPECTED_SEEDS)}).",
        r"\begin{itemize}",
    ]
    datasets = sorted(
        intra["dataset"].dropna().unique().tolist()
    )
    for ds in datasets:
        sub = intra[intra["dataset"] == ds].copy()
        if sub.empty:
            continue
        # Seleciona o modelo com maior F1 MÉDIO
        best_idx = sub["f1_macro"].idxmax()
        best     = sub.loc[best_idx]
        f1_mean  = best["f1_macro"]
        f1_std   = (
            best["f1_macro_std"]
            if "f1_macro_std" in best.index
            else None
        )
        n_seeds  = (
            int(best["n_seeds"])
            if "n_seeds" in best.index
            else len(EXPECTED_SEEDS)
        )
        val_str  = _fmt_mean_std(f1_mean, f1_std)
        # Ramos distintos: incompleto vs. completo
        seed_note = (
            f" ($n={n_seeds}$ seeds, incompleto)"
            if n_seeds < len(EXPECTED_SEEDS)
            else f" ($n={n_seeds}$ seeds)"
        )
        lines.append(
            f"  \\item \\textit{{{_ds_label(ds)}}}: "
            f"\\textbf{{{_model_label(best['model'])}}} "
            f"com F1-macro = {val_str}{seed_note}."
        )
    lines.append(        r"\end{itemize}"
    )
    lines.append("")
    content += "\n".join(lines) + "\n"
    return content
# ── Fragmento 10 — Sumário executivo ─────────────────────────────────────────
def gen_full_summary(
    df: pd.DataFrame,
    results_dir: Path,
) -> str:
    """
    Sumário executivo usando médias agregadas por seed.
    Não seleciona seed individual como melhor resultado.
    """
    content = _header_comment(
        "Sumário executivo — principais resultados"
    )
    intra = df[
        df["experiment_id"].str.startswith("EXP_A")
    ].copy()
    cross = df[
        df["experiment_id"].str.startswith("EXP_B")
    ].copy()
    lines = [
        r"\section*{Sumário dos Resultados Experimentais}",
        r"% Gerado automaticamente. "
        r"Inclua via \input{} no .tex principal.",
        "",
    ]
    n_experiments = len(df)
    n_datasets    = intra["dataset"].nunique()
    n_models      = intra["model"].nunique()
    lines += [
        r"\noindent\textbf{Visão geral:} "
        f"Foram realizadas {n_experiments} configurações "
        f"experimentais distribuídas em {n_models} modelos "
        f"e {n_datasets} datasets, cada uma com "
        f"{len(EXPECTED_SEEDS)} execuções independentes "
        f"(seeds {min(EXPECTED_SEEDS)}"
        f"--{max(EXPECTED_SEEDS)}).",
        "",
    ]
    # Melhor resultado em média — não em seed individual
    if not intra.empty and "f1_macro" in intra.columns:
        best_idx    = intra["f1_macro"].idxmax()
        best_global = intra.loc[best_idx]
        f1_mean     = best_global["f1_macro"]
        f1_std      = (
            best_global["f1_macro_std"]
            if "f1_macro_std" in best_global.index
            else None
        )
        val_str = _fmt_mean_std(f1_mean, f1_std)
        lines += [
            r"\noindent\textbf{Melhor resultado "
            r"intra-dataset (F1-macro médio):} "
            f"\\textbf{{{_model_label(best_global['model'])}}} "
            f"no dataset "
            f"\\textit{{{_ds_label(best_global['dataset'])}}} "
            f"com F1-macro = {val_str}.",
            "",
        ]
    # Generalização cross-dataset
    if not cross.empty and "f1_macro" in cross.columns:
        best_cross  = cross.loc[cross["f1_macro"].idxmax()]
        worst_cross = cross.loc[cross["f1_macro"].idxmin()]
        lines += [
            r"\noindent\textbf{Generalização "
            r"cross-dataset:} "
            f"O maior F1-macro médio cross-dataset foi "
            f"\\textbf{{{_fmt(best_cross['f1_macro'])}}} "
            f"({_model_label(best_cross['model'])} "
            f"--- EXP\\_B). "
            f"O menor foi {_fmt(worst_cross['f1_macro'])} "
            f"({_model_label(worst_cross['model'])}).",
            "",
        ]
    # Tabela rápida de médias por modelo
    if not intra.empty and "f1_macro" in intra.columns:
        avg_by_model = (
            intra.groupby("model")["f1_macro"]
            .mean()
            .sort_values(ascending=False)
        )
        lines += [
            r"\noindent\textbf{F1-macro médio "
            r"intra-dataset por modelo "
            r"(média das médias por dataset):}",
            "",
            r"\begin{tabular}{lc}",
            r"\hline "
            r"\textbf{Modelo} & "
            r"\textbf{F1-macro médio} \\ \hline",
        ]
        for model, f1 in avg_by_model.items():
            lines.append(
                f"  {_model_label(model)} & "
                f"{_fmt(f1)} \\\\"
            )
        lines.append(r"\hline \end{tabular}")
        lines.append("")
    content += "\n".join(lines) + "\n"
    return content
# ── Fragmento 11 — Tempo de treinamento ──────────────────────────────────────
def gen_train_time_table(df: pd.DataFrame) -> str:
    content = _header_comment(
        "Tabela: Tempo de Treinamento"
    )
    intra    = df[
        df["experiment_id"].str.startswith("EXP_A")
    ].copy()
    time_col: Optional[str] = None
    for candidate in ("train_time_s", "train_time_seconds"):
        if candidate in intra.columns:
            time_col = candidate
            break
    if time_col is None:
        return (
            content
            + "% Coluna de tempo de treinamento "
            + "não encontrada.\n"
        )
    time_df = intra[
        ["model", "dataset", time_col]
    ].dropna(subset=[time_col])
    if time_df.empty:
        return (
            content
            + "% Dados de tempo de treinamento "
            + "não disponíveis.\n"
        )
    datasets = sorted(
        time_df["dataset"].dropna().unique().tolist()
    )
    models   = [
        "logistic_regression", "mlp",
        "bilstm", "bertimbau",
    ]
    col_spec  = "l" + "c" * len(datasets)
    ds_header = " & ".join(
        [
            r"\textbf{" + _ds_label(d) + "}"
            for d in datasets
        ]
    )
    content += (
        r"\begin{table}[htbp]" + "\n"
        r"  \centering" + "\n"
        f"  \\caption{{Tempo médio de treinamento "
        f"(segundos) por modelo e dataset "
        f"(médias sobre {len(EXPECTED_SEEDS)} seeds).}}\n"
        r"  \label{tab:train_time}" + "\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        r"    \hline" + "\n"
        f"    \\textbf{{Modelo}} & {ds_header} \\\\\n"
        r"    \hline" + "\n"
    )
    for model in models:
        sub  = time_df[time_df["model"] == model]
        vals = []
        for ds in datasets:
            m_row = sub[sub["dataset"] == ds]
            if m_row.empty:
                vals.append("---")
            else:
                t = m_row[time_col].values[0]
                vals.append(
                    f"{float(t):.1f}s"
                    if t is not None
                    else "---"
                )
        content += (
            f"    {_model_label(model)} & "
            + " & ".join(vals)
            + " \\\\\n"
        )
    content += (
        r"    \hline" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )
    return content
# ── Fragmento 12 — Master file ────────────────────────────────────────────────
def gen_input_master(output_dir: Path) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"% master_report.tex — Gerado em {ts}",
        "% Inclua no seu .tex principal com:",
        r"%   \input{latex/report_fragments/master_report}",
        "",
        "% Resumo dos datasets",
        r"\input{latex/report_fragments/datasets_summary}",
        "",
        "% Experimento A — Intra-dataset",
        r"\input{latex/report_fragments/exp_A_intra}",
        r"\input{latex/report_fragments/exp_A_full_metrics}",
        "",
        "% Experimento B — Cross-dataset",
        r"\input{latex/report_fragments/exp_B_cross}",
        "",
        "% Experimento C — Combined",
        r"\input{latex/report_fragments/exp_C_combined}",
        "",
        "% Experimento D — Temporal",
        r"\input{latex/report_fragments/exp_D_temporal}",
        "",
        "% Experimento E — Robustez",
        r"\input{latex/report_fragments/exp_E_robustness}",
        "",
        "% Generalization Gap",
        r"\input{latex/report_fragments/generalization_gap}",
        "",
        "% Melhor modelo (parágrafo)",
        r"\input{latex/report_fragments/comparison_best_model}",
        "",
        "% Tempo de treinamento",
        r"\input{latex/report_fragments/train_time}",
        "",
        "% Sumário executivo completo",
        r"\input{latex/report_fragments/full_report_summary}",
    ]
    return "\n".join(lines) + "\n"
# ── Main ──────────────────────────────────────────────────────────────────────
def main(
    results_dir: Path = RESULTS_DIR,
    output_dir:  Path = OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("  Gerador de Relatório LaTeX — Monografia Fake News")
    print(f"  Resultados de: {results_dir}")
    print(f"  Saída em:      {output_dir}")
    print(f"{'='*60}\n")
    df = load_all_results(results_dir)
    print(
        f"  Carregados {len(df)} grupos de experimentos "
        f"(após agregação por seed)."
    )
    if df.empty:
        print(
            "  [AVISO] Nenhum resultado encontrado em "
            f"{results_dir}. "
            "Verifique se os experimentos foram executados."
        )
    fragments = [
        (
            "datasets_summary.tex",
            gen_datasets_summary(results_dir),
        ),
        (
            "exp_A_intra.tex",
            gen_exp_A_intra(df),
        ),
        (
            "exp_A_full_metrics.tex",
            gen_exp_A_full_metrics(df),
        ),
        (
            "exp_B_cross.tex",
            gen_exp_B_cross(df),
        ),
        (
            "exp_C_combined.tex",
            gen_exp_C_combined(df),
        ),
        (
            "exp_D_temporal.tex",
            gen_exp_D_temporal(df),
        ),
        (
            "exp_E_robustness.tex",
            gen_exp_E_robustness(results_dir),
        ),
        (
            "generalization_gap.tex",
            gen_generalization_gap(results_dir),
        ),
        (
            "comparison_best_model.tex",
            gen_best_model_paragraph(df),
        ),
        (
            "train_time.tex",
            gen_train_time_table(df),
        ),
        (
            "full_report_summary.tex",
            gen_full_summary(df, results_dir),
        ),
    ]
    for fname, content in fragments:
        fpath   = output_dir / fname
        fpath.write_text(content, encoding="utf-8")
        n_lines = content.count("\n")
        print(
            f"  [OK] {fname:<42} ({n_lines} linhas)"
        )
    master_path = output_dir / "master_report.tex"
    master_path.write_text(
        gen_input_master(output_dir),
        encoding="utf-8",
    )
    print(
        f"  [OK] {'master_report.tex':<42} "
        f"(arquivo mestre)"
    )
    print(
        f"\n  Fragmentos gerados em: {output_dir}"
    )
    print(
        "  Para incluir no Overleaf, adicione ao "
        ".tex principal:"
    )
    print(
        r"    \input{latex/report_fragments/master_report}"
    )
    print(f"\n{'='*60}\n")
# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Gera fragmentos LaTeX a partir dos "
            "JSONs de resultados."
        )
    )
    ap.add_argument(
        "--results",
        default=str(RESULTS_DIR),
        help=(
            f"Diretório com JSONs de resultados "
            f"(padrão: {RESULTS_DIR})"
        ),
    )
    ap.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help=(
            f"Diretório de saída para os .tex "
            f"(padrão: {OUTPUT_DIR})"
        ),
    )
    args = ap.parse_args()
    main(Path(args.results), Path(args.output))