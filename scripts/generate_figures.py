"""
scripts/generate_figures.py - Figuras para a monografia (6 datasets).
Correções aplicadas:
  - BERT (EN) e BERTimbau (PT) com rótulos e cores distintos.
  - load_intra() agrega por seed antes de plotar.
  - Legenda das figuras especifica a origem dos intervalos de erro.
  - fig_combined_vs_intra() não mistura os dois modelos BERT.
Datasets: Fake.br | FakeTrue.Br | FakeRecogna |
          FakeNewsNet | CONSTRAINT-COVID19 | PHEME
"""
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING = True
    sns.set_theme(style="whitegrid", font_scale=1.1)
    # 5 cores distintas - uma por modelo
    # Definido aqui para garantir que PAL esteja sempre disponível
    PAL = sns.color_palette("colorblind", 5)
except ImportError:
    plt = sns = None
    # Fallback: lista de None para que MODEL_COLORS seja sempre
    # definível, mesmo sem matplotlib/seaborn instalados
    PAL = [None] * 5
    PLOTTING = False
    print("[AVISO] Execute: pip install matplotlib seaborn")
RESULTS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
# Seeds esperadas - declarada como tupla para evitar mutação acidental
EXPECTED_SEEDS: Tuple[int, ...] = tuple(range(42, 52))  # 10 seeds: 42..51
# Rótulos distintos para BERT (EN) e BERTimbau (PT)
MODEL_LABELS: Dict[str, str] = {
    "logistic_regression": "LR + TF-IDF",
    "mlp":                 "MLP + TF-IDF",
    "bilstm":              "BiLSTM",
    "bert":                "BERT (EN)",
    "bertimbau":           "BERTimbau (PT)",
}
# Cores distintas para bert e bertimbau
MODEL_COLORS: Dict[str, object] = {
    "logistic_regression": PAL[0],
    "mlp":                 PAL[1],
    "bilstm":              PAL[2],
    "bert":                PAL[3],
    "bertimbau":           PAL[4],
}
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
MODEL_ORDER: List[str] = [
    "logistic_regression", "mlp", "bilstm",
    "bert", "bertimbau",
]
DATASET_ORDER: List[str] = [
    "fakebr", "faketruebr", "fakerecogna",
    "fakenewsnet", "constraint_covid19", "pheme",
]
# ── Helpers ───────────────────────────────────────────────────────────────────
def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(
            FIGURES_DIR / f"{stem}.{ext}",
            dpi=150,
            bbox_inches="tight",
        )
    print(f"[OK] {stem}.pdf/png")
    plt.close(fig)
def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
# ── Carregamento e agregação por seed ────────────────────────────────────────
def load_intra() -> pd.DataFrame:
    """
    Carrega resultados do EXP_A agrupando por seed.
    Cada linha do DataFrame representa a MÉDIA ± STD sobre as seeds
    de um mesmo par (model, dataset).
    Também extrai o IC de bootstrap (f1_macro_ci_lo/hi), média sobre as
    seeds com IC válido (NaN se nenhuma).
    Nota: esse IC mede a incerteza amostral do TESTE (modelo fixo), e
    NÃO a variabilidade de treino entre seeds (f1_macro_std). Os dois
    são colunas distintas e nunca devem ser rotulados como a mesma coisa.
    """
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for jf in sorted(RESULTS_DIR.glob("EXP_A_*.json")):
        d     = _load_json(jf)
        model = d.get("model", "")
        ds    = d.get("dataset", "")
        if not model or not ds:
            continue
        groups[(model, ds)].append(d)
    rows = []
    for (model, ds), entries in groups.items():
        f1_vals:  List[float] = []
        acc_vals: List[float] = []
        mcc_vals: List[float] = []
        tt_vals:  List[float] = []
        ci_lo_vals: List[float] = []
        ci_hi_vals: List[float] = []
        for e in entries:
            m = e.get("metrics", {})
            v = m.get("f1_macro")
            if v is not None:
                try:
                    f1_vals.append(float(v))
                except (TypeError, ValueError):
                    pass
            a = m.get("accuracy")
            if a is not None:
                try:
                    acc_vals.append(float(a))
                except (TypeError, ValueError):
                    pass
            mc = m.get("mcc")
            if mc is not None:
                try:
                    mcc_vals.append(float(mc))
                except (TypeError, ValueError):
                    pass
            tt = (
                m.get("train_time_s")
                or m.get("train_time_seconds")
            )
            if tt is not None:
                try:
                    tt_vals.append(float(tt))
                except (TypeError, ValueError):
                    pass
            # IC bootstrap por seed (ignora None/NaN)
            lo = m.get("f1_macro_ci_lo")
            hi = m.get("f1_macro_ci_hi")
            if lo is not None and hi is not None:
                try:
                    if (
                        np.isfinite(float(lo))
                        and np.isfinite(float(hi))
                    ):
                        ci_lo_vals.append(float(lo))
                        ci_hi_vals.append(float(hi))
                except (TypeError, ValueError):
                    pass
        if not f1_vals:
            continue
        f1_mean = float(np.mean(f1_vals))
        f1_std          = (
            float(np.std(f1_vals, ddof=1))
            if len(f1_vals) > 1
            else 0.0
        )
        n_seeds = len(f1_vals)
        if n_seeds < len(EXPECTED_SEEDS):
            print(
                f"[AVISO] load_intra: {model}/{ds}: "
                f"apenas {n_seeds} seed(s) encontrada(s) "
                f"(esperadas {len(EXPECTED_SEEDS)})."
            )
        rows.append({
            "model":          model,
            "dataset":        ds,
            "f1_macro":       f1_mean,
            "f1_macro_std":   f1_std,
            "f1_macro_ci_lo": (
                float(np.mean(ci_lo_vals))
                if ci_lo_vals
                else float("nan")
            ),
            "f1_macro_ci_hi": (
                float(np.mean(ci_hi_vals))
                if ci_hi_vals
                else float("nan")
            ),
            "accuracy":   (
                float(np.mean(acc_vals))
                if acc_vals else None
            ),
            "mcc":        (
                float(np.mean(mcc_vals))
                if mcc_vals else None
            ),
            "train_time": (
                float(np.mean(tt_vals))
                if tt_vals else None
            ),
            "n_seeds":    n_seeds,
        })
    return pd.DataFrame(rows)
# ── Fig 1: F1-macro intra-dataset ────────────────────────────────────────────
def fig_f1_comparison(df: pd.DataFrame) -> None:
    """
    Barras agrupadas: F1-macro médio por modelo e dataset.
    Barras de erro: desvio padrão entre seeds (não IC bootstrap).
    A legenda especifica explicitamente que as barras de erro
    representam ± 1 desvio padrão sobre as seeds disponíveis.
    """
    if df.empty or not PLOTTING:
        return
    datasets = [
        d for d in DATASET_ORDER
        if d in df["dataset"].values
    ]
    models = [
        m for m in MODEL_ORDER
        if m in df["model"].values
    ]
    if not datasets or not models:
        print(
            "[AVISO] fig_f1_comparison: "
            "sem dados suficientes."
        )
        return
    x = np.arange(len(datasets))
    w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(models):
        sub = (
            df[df["model"] == model]
            .set_index("dataset")
        )
        vals    = []
        errs_lo = []
        errs_hi = []
        for d in datasets:
            if d not in sub.index:
                vals.append(0)
                errs_lo.append(0)
                errs_hi.append(0)
                continue
            row   = sub.loc[d]
            # Garante escalar mesmo se houver múltiplas linhas
            f1    = float(
                pd.Series(row["f1_macro"]).mean()
            )
            f1std = float(
                pd.Series(row["f1_macro_std"]).mean()
            )
            vals.append(f1)
            errs_lo.append(f1std)
            errs_hi.append(f1std)
        ax.bar(
            x + i * w - (len(models) - 1) * w / 2,
            vals,
            w,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "gray"),
            yerr=[errs_lo, errs_hi],
            capsize=3,
            error_kw={"linewidth": 1},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_LABELS.get(d, d) for d in datasets],
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "F1-macro por Modelo e Dataset - "
        "Exp. A: Intra-dataset\n"
        r"(barras de erro: $\pm$1 desvio padrão entre seeds)"
    )
    ax.legend(
        title="Modelo",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    fig.tight_layout()
    _save(fig, "f1_comparison")
# ── Fig 2: Heatmap cross-dataset ─────────────────────────────────────────────
def fig_cross_heatmap() -> None:
    """
    Heatmap F1-macro médio para cada par (train, test).
    Usa a coluna f1_macro_mean quando disponível; caso contrário,
    usa f1_macro (média de compatibilidade retroativa).
    Apenas médias são exibidas - variabilidade entre seeds não é
    representável em heatmap; indicar nas notas da tabela correspondente.
    """
    csv_p = RESULTS_DIR / "cross_dataset_summary.csv"
    if not csv_p.exists() or not PLOTTING:
        return
    df     = pd.read_csv(csv_p)
    models = (
        [m for m in MODEL_ORDER if m in df["model"].values]
        or sorted(df["model"].unique())
    )
    value_col = (
        "f1_macro_mean"
        if "f1_macro_mean" in df.columns
        else "f1_macro"
    )
    for model in models:
        sub = df[df["model"] == model].copy()
        sub["train_label"] = sub["train_dataset"].map(
            lambda ds: DATASET_LABELS.get(ds, ds)
        )
        sub["test_label"] = sub["test_dataset"].map(
            lambda ds: DATASET_LABELS.get(ds, ds)
        )
        pivot = sub.pivot_table(
            index="train_label",
            columns="test_label",
            values=value_col,
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd_r",
            vmin=0,
            vmax=1,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "F1-macro (média)"},
        )
        ax.set_title(
            f"Cross-dataset F1-macro - "
            f"{MODEL_LABELS.get(model, model)}\n"
            f"(Exp. B - médias sobre seeds)"
        )
        fig.tight_layout()
        _save(fig, f"cross_dataset_heatmap_{model}")
    print(
        f"[OK] cross_dataset heatmaps "
        f"({len(models)} modelos)"
    )
# ── Fig 3: Generalization Gap ─────────────────────────────────────────────────
def fig_gap() -> None:
    p = RESULTS_DIR / "generalization_gap.csv"
    if not p.exists() or not PLOTTING:
        return
    df = pd.read_csv(p)
    # gap_between_ds_std: desvio ENTRE datasets de treino
    # (heterogeneidade de generalização por modelo
        # - distinto de generalization_gap_std, que é o desvio
    # ENTRE seeds para um mesmo dataset de treino).
    avg = df.groupby("model", as_index=False).agg(
        generalization_gap=("generalization_gap", "mean"),
        gap_between_ds_std=("generalization_gap", "std"),
    )
    avg["gap_between_ds_std"] = (
        avg["gap_between_ds_std"].fillna(0.0)
    )
    avg = avg.sort_values(
        "generalization_gap", ascending=False
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        range(len(avg)),
        avg["generalization_gap"],
        yerr=avg["gap_between_ds_std"],
        capsize=4,
        color=[
            MODEL_COLORS.get(m, "gray")
            for m in avg["model"]
        ],
    )
    ax.set_xticks(range(len(avg)))
    ax.set_xticklabels(
        [
            MODEL_LABELS.get(m, m)
            for m in avg["model"]
        ],
        rotation=10,
    )
    ax.set_ylabel("Gap (F1-intra − F1-cross médio)")
    ax.set_title(
        "Generalization Gap por Modelo\n"
        "(gap pareado por seed; barra = desvio entre "
        "datasets de treino)"
    )
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    fig.tight_layout()
    _save(fig, "generalization_gap")
# ── Fig 4: Robustez ───────────────────────────────────────────────────────────
def fig_robustness() -> None:
    files = sorted(RESULTS_DIR.glob("robustness_*.csv"))
    if not files or not PLOTTING:
        return
    df = pd.concat(
        [pd.read_csv(f) for f in files],
        ignore_index=True,
        sort=False,
    )
    df = df[df["perturbation"] != "baseline"].copy()
    # Usa delta_f1 (média entre seeds já calculada no CSV)
    avg = df.groupby(
        ["model", "perturbation"], as_index=False
    )["delta_f1"].mean()
    pivot = avg.pivot(
        index="perturbation",
        columns="model",
        values="delta_f1",
    )
    pivot.columns = [
        MODEL_LABELS.get(c, c) for c in pivot.columns
    ]
    fig, ax = plt.subplots(figsize=(15, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Perturbação")
    ax.set_ylabel(
        r"$\Delta$F1 (baseline $-$ perturbado)"
    )
    ax.set_title(
        "Robustez: impacto das perturbações linguísticas "
        "(Exp. E)\n"
        r"(positivo = degradação; médias sobre seeds)"
    )
    ax.legend(
        title="Modelo",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    ax.axhline(0, color="gray", linestyle="--", lw=0.8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, "robustness")
# ── Fig 5: Tempo de treinamento ───────────────────────────────────────────────
def fig_train_time(df: pd.DataFrame) -> None:
    if df.empty or not PLOTTING:
        return
    td = df[["model", "dataset", "train_time"]].dropna()
    if td.empty:
        print(
            "[AVISO] fig_train_time: "
            "nenhum dado de tempo disponível."
        )
        return
    pivot = td.pivot_table(
        index="model",
        columns="dataset",
        values="train_time",
        aggfunc="mean",
    )
    pivot.index = [
        MODEL_LABELS.get(i, i) for i in pivot.index
    ]
    pivot.columns = [
        DATASET_LABELS.get(c, c) for c in pivot.columns
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Tempo médio (s)")
    ax.set_title(
        "Tempo de Treinamento por Modelo e Dataset\n"
        "(médias sobre seeds)"
    )
    ax.legend(
        title="Dataset",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=10
    )
    fig.tight_layout()
    _save(fig, "train_time")
# ── Fig 6: Combined vs Intra ──────────────────────────────────────────────────
def fig_combined_vs_intra(
    intra_df: pd.DataFrame,
    language: str,
) -> None:
    """
    Compara F1-macro do treinamento combinado (EXP_C) com o
    treinamento intra-dataset (EXP_A) para cada dataset de teste.
    Não remove prefixo de idioma do nome do modelo para evitar
    confundir BERT (EN) com BERTimbau (PT).
    Usa coluna f1_macro_mean quando disponível.
    Filtra por coluna 'language' de forma robusta.
    """
    cp = RESULTS_DIR / "combined_summary.csv"
    if not cp.exists() or not PLOTTING or intra_df.empty:
        return
    cdf = pd.read_csv(cp).copy()
    if cdf.empty:
        return
    # Filtra por idioma usando a coluna 'language'
    if "language" in cdf.columns:
        lang_col = (
            cdf["language"]
            .astype(str)
            .str.lower()
            .str.strip()
        )
        if language == "pt":
            cdf = cdf[
                lang_col.isin([
                    "pt", "pt-br",
                    "portuguese", "português",
                ])
            ]
        else:
            cdf = cdf[
                lang_col.isin([
                    "en", "en-us",
                    "english", "inglês",
                ])
            ]
    elif "test_dataset" in cdf.columns:
        prefix = f"combined_{language}-"
        cdf = cdf[
            cdf["test_dataset"]
            .astype(str)
            .str.startswith(prefix)
        ]
    if cdf.empty:
        print(
            f"[AVISO] Sem resultados combinados "
            f"para {language.upper()}."
        )
        return
    metric_col = (
        "f1_macro_mean"
        if "f1_macro_mean" in cdf.columns
        else "f1_macro"
    )
    # Extrai chave do dataset de teste
    cdf["dataset_key"] = (
        cdf["test_dataset"]
        .astype(str)
        .str.replace(
            r"^combined_(pt|en)-", "", regex=True
        )
    )
    models = [
        m for m in MODEL_ORDER
        if m in cdf["model"].values
    ]
    if not models        :
        print(
            f"[AVISO] Nenhum modelo reconhecido "
            f"para {language.upper()}."
        )
        return
    test_datasets = list(
        dict.fromkeys(cdf["dataset_key"].tolist())
    )
    x = np.arange(len(test_datasets))
    w = 0.8 / max(len(models) * 2, 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(models):
        sub_i = (
            intra_df[intra_df["model"] == model]
            .copy()
            .groupby("dataset")["f1_macro"]
            .mean()
        )
        sub_c = (
            cdf[cdf["model"] == model]
            .copy()
            .groupby("dataset_key")[metric_col]
            .mean()
        )
        v_intra = [
            float(sub_i[d]) if d in sub_i.index
            else float("nan")
            for d in test_datasets
        ]
        v_comb = [
            float(sub_c[d]) if d in sub_c.index
            else float("nan")
            for d in test_datasets
        ]
        ml = MODEL_LABELS.get(model, model)
        # Centraliza os 2*len(models) grupos de barras
        # em torno de cada tick.
        # Total de colunas por dataset: 2 * len(models)
        # Offset para centralizar: (len(models) - 0.5) * w
        offset = (len(models) - 0.5) * w
        ax.bar(
            x + (2 * i) * w - offset,
            v_intra,
            w,
            label=f"{ml} intra",
            color=MODEL_COLORS.get(model, "gray"),
            alpha=0.5,
        )
        ax.bar(
            x + (2 * i + 1) * w - offset,
            v_comb,
            w,
            label=f"{ml} combined",
            color=MODEL_COLORS.get(model, "gray"),
            alpha=1.0,
            hatch="//",
        )
    labels = [
        DATASET_LABELS.get(d, d) for d in test_datasets
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1.15)
    language_label = (
        "PT-BR" if language == "pt" else "inglês"
    )
    ax.set_title(
        f"Combined {language_label} vs. Intra-dataset "
        f"(Exp. C)\n"
        f"(médias sobre seeds; "
        f"tracejado = treinamento combinado)"
    )
    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, f"combined_{language}_vs_intra")
# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not PLOTTING:
        print(
            "[ERRO] matplotlib/seaborn necessários.\n"
            "       Execute: pip install matplotlib seaborn"
        )
        return
    df = load_intra()
    if df.empty:
        print(
            "[AVISO] Nenhum resultado EXP_A encontrado. "
            "Algumas figuras não serão geradas."
        )
    fig_f1_comparison(df)
    fig_cross_heatmap()
    fig_gap()
    fig_robustness()
    fig_train_time(df)
    fig_combined_vs_intra(df, "pt")
    fig_combined_vs_intra(df, "en")
    print(
        f"\n[PRONTO] Figuras em "
        f"{FIGURES_DIR.resolve()}/"
    )
if __name__ == "__main__":
    main()