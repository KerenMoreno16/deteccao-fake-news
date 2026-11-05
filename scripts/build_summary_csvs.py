# -*- coding: utf-8 -*-
"""
scripts/build_summary_csvs.py
Constrói os CSVs auxiliares a partir dos EXP_*.json em results/metrics/.
Versão 6.0 — correções aplicadas:
  - Diferenciação entre BERT (EN) e BERTimbau (PT).
  - Detecção e alerta de seeds faltantes.
  - Prevenção de duplicação por arquivos legacy sem sufixo de seed.
  - Cálculo correto do generalization gap (sem média de médias não ponderada).
  - Mensagens de diagnóstico mais informativas.
Para cada tipo de experimento, os arquivos individuais por seed são
agrupados e a média ± desvio padrão (ddof=1) é calculada.
Saídas em results/metrics/:
  cross_dataset_summary.csv      — (model, train_ds, test_ds, mean ± std)
  generalization_gap.csv         — F1_intra − F1_cross médio
  robustness_{model}.csv         — ΔF1 por perturbação
  combined_summary.csv           — F1 combined PT-BR / EN
  temporal_summary.csv           — F1 split temporal
"""
from pathlib import Path
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
METRICS_DIR = Path("results/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)
# Seeds esperadas — usadas para alertar sobre execuções faltantes.
# Declarada como tupla para evitar mutação acidental.
EXPECTED_SEEDS: Tuple[int, ...] = tuple(range(42, 52))  # 10 seeds: 42..51
DATASET_DISPLAY: Dict[str, str] = {
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
MODEL_DISPLAY: Dict[str, str] = {
    "logistic_regression": "LR + TF-IDF",
    "mlp":                 "MLP + TF-IDF",
    "bilstm":              "BiLSTM",
    "bert":                "BERT (EN)",
    "bertimbau":           "BERTimbau (PT)",
}
def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
def _ds(n: str) -> str:
    return DATASET_DISPLAY.get(n, n)
def _m(n: str) -> str:
    return MODEL_DISPLAY.get(n, n)
def _mean_std(
    values: List[float],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Retorna (média, desvio padrão amostral ddof=1).
    Retorna (None, None) se a lista estiver vazia.
    Retorna (valor, 0.0) se houver apenas um elemento.
    """
    arr = [v for v in values if v is not None]
    if not arr:
        return None, None
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std
def _fmt(
    mean: Optional[float],
    std: Optional[float],
    decimals: int = 4,
) -> str:
    if mean is None:
        return "---"
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
def _check_seeds(
    files: List[Path],
    base_id: str = "",
    expected: Optional[Tuple[int, ...]] = None,
) -> None:
    """
    Emite aviso se alguma seed esperada estiver ausente no grupo.
    Também alerta sobre arquivos duplicados (com e sem sufixo de seed).
    """
    if expected is None:
        expected = EXPECTED_SEEDS
    found: Set[int] = set()
    for f in files:
        m = re.search(r"_seed(\d+)$", f.stem)
        if m:
            found.add(int(m.group(1)))
    missing = set(expected) - found
    if missing:
        print(
            f"[AVISO] {base_id}: seeds faltantes: "
            f"{sorted(missing)} "
            f"(encontradas: {sorted(found)})"
        )
def _group_by_base(
    pattern: str,
) -> Dict[str, List[Path]]:
    """
    Varre METRICS_DIR buscando arquivos que correspondam ao padrão
    e os agrupa pelo ID base (sem sufixo _seed{N}).
    Arquivos sem sufixo de seed são incluídos apenas se não existir
    nenhum arquivo com sufixo para o mesmo ID base — evita duplicatas.
    """
    with_seed:    Dict[str, List[Path]] = defaultdict(list)
    without_seed: Dict[str, List[Path]] = defaultdict(list)
    for f in sorted(METRICS_DIR.glob(pattern)):
        stem = f.stem
        if re.search(r"_seed\d+$", stem):
            base = re.sub(r"_seed\d+$", "", stem)
            with_seed[base].append(f)
        else:
            without_seed[stem].append(f)
    groups: Dict[str, List[Path]] = defaultdict(list)
    for base, files in with_seed.items():
        groups[base].extend(files)
    for base, files in without_seed.items():
        if base in groups:
            print(
                f"[AVISO] Arquivo(s) legacy sem sufixo de seed "
                f"ignorado(s) para '{base}': "
                f"{[f.name for f in files]}. "
                f"Já existem arquivos com seed para esse ID."
            )
        else:
            groups[base].extend(files)
    for base in groups:
        def _sort_key(p: Path) -> int:
            m = re.search(r"_seed(\d+)$", p.stem)
            return int(m.group(1)) if m else -1
        groups[base].sort(key=_sort_key)
    return dict(groups)
# ── 1. cross_dataset_summary.csv ─────────────────────────────────────────────
def build_cross_dataset() -> pd.DataFrame:
    """
    Agrega resultados do EXP_B por (model, train_dataset, test_dataset)
    calculando média e desvio padrão sobre as seeds disponíveis.
    Detecta e alerta sobre arquivos legacy duplicados.
    """
    pat_str = re.compile(
        r"EXP_B_(?P<model>.+?)_TRAIN_(?P<train>.+?)_"
                r"TEST_(?P<test>.+?)_seed(?P<seed>\d+)$"
    )
    pat_legacy = re.compile(
        r"EXP_B_(?P<model>.+?)_TRAIN_(?P<train>.+?)_TEST_(?P<test>.+?)$"
    )
    with_seed:    Dict[Tuple[str, str, str], List[Path]] = defaultdict(list)
    without_seed: Dict[Tuple[str, str, str], List[Path]] = defaultdict(list)
    seen_paths:   Set[str] = set()
    for f in sorted(METRICS_DIR.glob("EXP_B_*.json")):
        canonical = str(f.resolve())
        if canonical in seen_paths:
            continue
        seen_paths.add(canonical)
        stem = f.stem
        m_seed = pat_str.match(stem)
        if m_seed:
            key = (
                m_seed.group("model"),
                m_seed.group("train"),
                m_seed.group("test"),
            )
            with_seed[key].append(f)
        else:
            clean = re.sub(r"_seed\d+$", "", stem)
            m_leg = pat_legacy.match(clean)
            if m_leg:
                key = (
                    m_leg.group("model"),
                    m_leg.group("train"),
                    m_leg.group("test"),
                )
                without_seed[key].append(f)
    groups: Dict[Tuple[str, str, str], List[Path]] = {}
    for key, files in with_seed.items():
        groups[key] = files
    for key, files in without_seed.items():
        if key in groups:
            print(
                f"[AVISO] EXP_B legacy ignorado para {key}: "
                f"{[f.name for f in files]}. "
                f"Já existem {len(groups[key])} arquivo(s) com seed."
            )
        else:
            groups[key] = files
    rows = []
    for (model, train, test), files in groups.items():
        base_id = f"EXP_B_{model}_TRAIN_{train}_TEST_{test}"
        _check_seeds(files, base_id=base_id)
        metrics_per_seed: Dict[str, List[float]] = defaultdict(list)
        for f in files:
            d   = _load(f)
            met = d.get("metrics", {})
            for metric_name in ("f1_macro", "accuracy", "mcc"):
                v = met.get(metric_name)
                if v is not None:
                    try:
                        metrics_per_seed[metric_name].append(float(v))
                    except (TypeError, ValueError):
                        pass
            # js_divergence_unigrams está na raiz do JSON
            v_js = d.get("js_divergence_unigrams")
            if v_js is not None:
                try:
                    metrics_per_seed[
                        "js_divergence_unigrams"
                    ].append(float(v_js))
                except (TypeError, ValueError):
                    pass
        row: Dict[str, Any] = {
            "model":          model,
            "model_display":  _m(model),
            "train_dataset":  train,
            "train_display":  _ds(train),
            "test_dataset":   test,
            "test_display":   _ds(test),
            "n_seeds":        len(files),
        }
        for k, values in metrics_per_seed.items():
            mean, std = _mean_std(values)
            row[f"{k}_mean"]      = mean
            row[f"{k}_std"]       = std
            row[f"{k}_formatted"] = _fmt(mean, std)
        # Colunas de compatibilidade retroativa (valor = média)
        for k in (
            "f1_macro", "accuracy", "precision_macro",
            "recall_macro", "mcc", "roc_auc",
        ):
            row[k] = row.get(f"{k}_mean")
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["model", "train_dataset", "test_dataset"]
        ).reset_index(drop=True)
        df.to_csv(
            METRICS_DIR / "cross_dataset_summary.csv",
            index=False,
        )
        print(
            f"[OK] cross_dataset_summary.csv  "
            f"({len(df)} linhas)"
        )
    else:
        print("[AVISO] Nenhum EXP_B_* encontrado.")
    return df
# ── 2. generalization_gap.csv ─────────────────────────────────────────────────
def _t_ci95(
    values: List[float],
) -> Tuple[Optional[float], Optional[float]]:
    """IC 95% t de Student para a média (n pequeno). (None, None) se n<2."""
    n = len(values)
    if n < 2:
        return None, None
    from scipy import stats
    mean = float(np.mean(values))
    sem  = float(np.std(values, ddof=1)) / np.sqrt(n)
    half = float(stats.t.ppf(0.975, n - 1)) * sem
    return mean - half, mean + half
def build_generalization_gap(cross_df: pd.DataFrame) -> None:
    """
    Generalization gap PAREADO por seed:
        gap_s = F1_intra(s) − média_{t ≠ train} F1_cross(s, t)
    Para cada (modelo, dataset de treino) e para cada seed s, o F1 intra
    (EXP_A, mesma seed) é comparado com o F1 médio nos demais corpora de
    teste (EXP_B, MESMA seed). O gap reportado é a média ± desvio de
    gap_s sobre as seeds, com IC 95% (t de Student). Só entram seeds
    presentes em AMBOS os experimentos (pareamento).
    gap_rel = gap / F1_intra (perda relativa).
    A média cross não é ponderada pelo tamanho dos corpora de teste.
    """
    pat_b = re.compile(
        r"EXP_B_(?P<model>.+?)_TRAIN_(?P<train>.+?)_TEST_(?P<test>.+?)"
        r"_seed(?P<seed>\d+)$"
    )
    # EXP_A: (model, dataset, seed) -> f1
    intra: Dict[Tuple[str, str, int], float] = {}
    for f in sorted(METRICS_DIR.glob("EXP_A_*.json")):
        d     = _load(f)
        model = d.get("model", "")
        ds    = d.get("dataset", "")
        f1    = d.get("metrics", {}).get("f1_macro")
        m     = re.search(r"_seed(\d+)$", f.stem)
        if not model or not ds or f1 is None or not m:
            continue
        try:
            intra[(model, ds, int(m.group(1)))] = float(f1)
        except (TypeError, ValueError):
            pass
    # EXP_B: (model, train, seed) -> {test: f1}, excluindo test == train
    cross: Dict[
        Tuple[str, str, int], Dict[str, float]
    ] = defaultdict(dict)
    for f in sorted(METRICS_DIR.glob("EXP_B_*.json")):
        m = pat_b.match(f.stem)
        if not m or m.group("train") == m.group("test"):
            continue
        f1 = _load(f).get("metrics", {}).get("f1_macro")
        if f1 is None:
            continue
        try:
            cross[
                (m.group("model"), m.group("train"), int(m.group("seed")))
            ][m.group("test")] = float(f1)
        except (TypeError, ValueError):
            pass
    if not intra or not cross:
        print(
            "[AVISO] EXP_A ou EXP_B ausentes — "
            "generalization_gap.csv não será gerado."
        )
        return
    rows = []
    pairs = sorted({(mo, ds) for (mo, ds, _s) in intra})
    for model, ds in pairs:
        seeds = sorted(
            s for (mo, d_, s) in intra
            if mo == model
            and d_ == ds
            and (model, ds, s) in cross
        )
        if not seeds:
            print(
                f"[AVISO] Sem seeds pareadas EXP_A/EXP_B para "
                f"model={model}, train={ds} — ignorado."
            )
            continue
        f1_in = [intra[(model, ds, s)] for s in seeds]
        f1_cr = [
            float(np.mean(list(cross[(model, ds, s)].values())))
            for s in seeds
        ]
        gaps = [a - b for a, b in zip(f1_in, f1_cr)]
        rel  = [
            g / a for g, a in zip(gaps, f1_in) if a > 0
        ]
        in_mean,  in_std  = _mean_std(f1_in)
        f1_cr_mean, f1_cr_std = _mean_std(f1_cr)
        gap_mean, gap_std = _mean_std(gaps)
        ci_lo, ci_hi      = _t_ci95(gaps)
        rows.append({
            "model":                        model,
            "model_display":                _m(model),
            "dataset":                      ds,
            "dataset_display":              _ds(ds),
            "f1_intra":                     in_mean,
            "f1_intra_std":                 in_std,
            "f1_intra_n_seeds":             len(seeds),
            "f1_intra_formatted":           _fmt(in_mean, in_std),
            "f1_cross_mean":                f1_cr_mean,
            "f1_cross_std":                 f1_cr_std,
            "f1_cross_n_pairs":             int(np.mean([
                len(cross[(model, ds, s)]) for s in seeds
            ])),
            "generalization_gap":           round(gap_mean, 4),
            "generalization_gap_std":       gap_std,
            "generalization_gap_ci95_low":  ci_lo,
            "generalization_gap_ci95_high": ci_hi,
            "generalization_gap_relative":  (
                float(np.mean(rel)) if rel else None
            ),
            "n_seeds_paired":               len(seeds),
        })
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(
            ["model", "dataset"]
        ).reset_index(drop=True)
        df.to_csv(
            METRICS_DIR / "generalization_gap.csv",
            index=False,
        )
        print(
            f"[OK] generalization_gap.csv  "
            f"({len(df)} linhas)"
        )
        print(
            "  NOTA: gap pareado por seed (EXP_A vs EXP_B, mesma "
            "seed); F1_cross é média não ponderada dos corpora de teste."
        )
    else:
        print(
            "[AVISO] Nenhuma linha gerada para "
            "generalization_gap.csv"
        )
# ── 3. robustness_{model}.csv ─────────────────────────────────────────────────
def build_robustness() -> None:
    """
    Agrega resultados do EXP_E por (model, dataset, perturbation)
    calculando média e desvio padrão sobre as seeds disponíveis.
    Usa _group_by_base() para evitar duplicatas de arquivos legacy.
    """
    groups_by_base = _group_by_base("EXP_E_*.json")
    if not groups_by_base:
        print("[AVISO] Nenhum EXP_E_* encontrado.")
        return
    agg: Dict[
        Tuple[str, str, str],
        List[Tuple[float, float]],
    ] = defaultdict(list)
    for base, files in groups_by_base.items():
        _check_seeds(files, base_id=base)
        for f in files:
            d       = _load(f)
            model   = d.get("model", "")
            dataset = d.get("dataset", "")
            met     = d.get("metrics", {})
            if not model or not dataset:
                print(
                    f"[AVISO] {f.name}: campos 'model' ou "
                    f"'dataset' ausentes — ignorado."
                )
                continue
            for pn, v in met.items():
                if not isinstance(v, dict):
                    continue
                f1    = v.get("f1_macro")
                delta = v.get("delta_f1")
                if f1 is not None and delta is not None:
                    try:
                        agg[(model, dataset, pn)].append(
                            (float(f1), float(delta))
                        )
                    except (TypeError, ValueError):
                        pass
    if not agg:
        print("[AVISO] Nenhum EXP_E_* com dados válidos encontrado.")
        return
    model_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (model, dataset, pn), pairs in agg.items():
        f1_vals    = [p[0] for p in pairs]
        delta_vals = [p[1] for p in pairs]
        f1_mean,    f1_std    = _mean_std(f1_vals)
        delta_mean, delta_std = _mean_std(delta_vals)
        model_rows[model].append({
            "model":           model,
            "model_display":   _m(model),
            "dataset":         dataset,
            "dataset_display": _ds(dataset),
            "perturbation":    pn,
            "n_seeds":         len(pairs),
            "f1_macro":        f1_mean,
            "f1_macro_std":    f1_std,
            "delta_f1":        delta_mean,
            "delta_f1_std":    delta_std,
            "f1_formatted":    _fmt(f1_mean, f1_std),
            "delta_formatted": _fmt(delta_mean, delta_std),
        })
    for model, rows in model_rows.items():
        df = pd.DataFrame(rows)
        df = df.sort_values(
            ["dataset", "perturbation"]
        ).reset_index(drop=True)
        out = METRICS_DIR / f"robustness_{model}.csv"
        df.to_csv(out, index=False)
        print(
            f"[OK] robustness_{model}.csv  "
            f"({len(df)} linhas)"
        )
# ── 4. combined_summary.csv ───────────────────────────────────────────────────
def build_combined_summary() -> None:
    """
    Agrega resultados do EXP_C por (lang, model, train_lang, test_dataset)
    calculando média e desvio padrão sobre as seeds disponíveis.
    Padrão esperado do nome do arquivo:
        EXP_C_{lang}_{model}_combined_{train_lang}_test_{test}_seed{N}.json
    Exemplo:
        EXP_C_pt_bertimbau_combined_pt_test_faketruebr_seed42.json
    """
    pat = re.compile(
        r"EXP_C_(?P<lang>pt|en)_(?P<model>.+?)_"
        r"combined_(?P<train_lang>pt|en)_test_"
        r"(?P<test>.+?)(?:_seed(?P<seed>\d+))?$"
    )
    groups: Dict[
        Tuple[str, str, str, str],
        List[Path],
    ] = defaultdict(list)
    unmatched: List[str] = []
    for f in sorted(METRICS_DIR.glob("EXP_C_*.json")):
        m = pat.match(f.stem)
        if not m:
            unmatched.append(f.name)
            continue
        key = (
            m.group("lang"),
            m.group("model"),
            m.group("train_lang"),
            m.group("test"),
        )
        groups[key].append(f)
    if unmatched:
        print(
            f"[AVISO] {len(unmatched)} arquivo(s) EXP_C não "
            f"corresponderam ao padrão esperado:\n"
            + "\n".join(f"  - {n}" for n in unmatched[:10])
            + ("\n  ..." if len(unmatched) > 10 else "")
        )
    # Separar arquivos com e sem seed dentro de cada grupo
    groups_with_seed:    Dict[
        Tuple[str, str, str, str], List[Path]
    ] = defaultdict(list)
    groups_without_seed: Dict[
        Tuple[str, str, str, str], List[Path]
    ] = defaultdict(list)
    for key, files in groups.items():
        for f in files:
            if re.search(r"_seed\d+$", f.stem):
                groups_with_seed[key].append(f)
            else:
                groups_without_seed[key].append(f)
    final_groups: Dict[
        Tuple[str, str, str, str], List[Path]
    ] = {}
    for key, files in groups_with_seed.items():
        final_groups[key] = files
    for key, files in groups_without_seed.items():
        if key in final_groups:
            print(
                f"[AVISO] EXP_C legacy ignorado para {key}: "
                f"{[f.name for f in files]}. "
                f"Já existem arquivos com seed para esse ID."
            )
        else:
            final_groups[key] = files
    rows = []
    for (lang, model, train_lang, test), files in (
        final_groups.items()
    ):
        base_id = (
            f"EXP_C_{lang}_{model}_combined_"
            f"{train_lang}_test_{test}"
        )
        _check_seeds(files, base_id=base_id)
        metrics_per_seed: Dict[str, List[float]] = defaultdict(list)
        seeds_found: Set[int] = set()
        for f in files:
            seed_match = re.search(r"_seed(\d+)$", f.stem)
            if seed_match:
                seeds_found.add(int(seed_match.group(1)))
            d   = _load(f)
            met = d.get("metrics", {})
            for metric_name in (
                "f1_macro", "accuracy", "precision_macro",
                "recall_macro", "mcc", "roc_auc",
            ):
                value = met.get(metric_name)
                if value is not None:
                    try:
                        metrics_per_seed[metric_name].append(
                            float(value)
                        )
                    except (TypeError, ValueError):
                        pass
        row: Dict[str, Any] = {
            "language":       lang,
            "model":          model,
            "model_display":  _m(model),
            "train_language": train_lang,
            "test_dataset":   test,
            "test_display":   _ds(test),
            "n_seeds":        len(seeds_found),
            "seeds":          ",".join(
                map(str, sorted(seeds_found))
            ),
        }
        for metric_name, values in metrics_per_seed.items():
            mean, std = _mean_std(values)
            row[f"{metric_name}_mean"]      = mean
            row[f"{metric_name}_std"]       = std
            row[f"{metric_name}_formatted"] = _fmt(mean, std)
        # Compatibilidade retroativa
        row["f1_macro"] = row.get("f1_macro_mean")
        row["accuracy"] = row.get("accuracy_mean")
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(
            ["language", "model", "test_dataset"]
        ).reset_index(drop=True)
        output = METRICS_DIR / "combined_summary.csv"
        df.to_csv(output, index=False)
        print(
            f"[OK] combined_summary.csv  "
            f"({len(df)} linhas)"
        )
        print(
            f"  Resultados PT: "
            f"{(df['language'] == 'pt').sum()} linhas"
        )
        print(
            f"  Resultados EN: "
            f"{(df['language'] == 'en').sum()} linhas"
        )
    else:
        print(
            "[AVISO] Nenhum EXP_C_* compatível encontrado. "
            "Verifique se os nomes dos arquivos seguem o padrão:\n"
            "  EXP_C_{lang}_{model}_combined_{train_lang}"
            "_test_{test}_seed{N}.json"
        )
# ── 5. temporal_summary.csv ───────────────────────────────────────────────────
def build_temporal_summary() -> None:
    """
    Agrega resultados do EXP_D por (model, dataset)
    calculando média e desvio padrão sobre as seeds disponíveis.
    Inclui js_divergence_unigrams quando presente nos JSONs.
    Alerta quando essa métrica está ausente (indica que
    calcular_jsd_classes ainda não foi integrado ao pipeline EXP_D).
    """
    pat = re.compile(
        r"EXP_D_(?P<model>.+?)_(?P<dataset>.+?)"
        r"_TEMPORAL(?:_seed(?P<seed>\d+))?$"
    )
    pat_legacy = re.compile(
        r"EXP_D_(?P<model>.+?)_(?P<dataset>.+?)"
        r"_temporal(?:_seed(?P<seed>\d+))?$"
    )
    with_seed:    Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    without_seed: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for f in sorted(METRICS_DIR.glob("EXP_D_*.json")):
        m = pat.match(f.stem) or pat_legacy.match(f.stem)
        if not m:
            continue
        key = (m.group("model"), m.group("dataset"))
        if re.search(r"_seed\d+$", f.stem):
            with_seed[key].append(f)
        else:
            without_seed[key].append(f)
    groups: Dict[Tuple[str, str], List[Path]] = {}
    for key, files in with_seed.items():
        groups[key] = files
    for key, files in without_seed.items():
        if key in groups:
            print(
                f"[AVISO] EXP_D legacy ignorado para {key}: "
                f"{[f.name for f in files]}."
            )
        else:
            groups[key] = files
    jsd_missing_count = 0
    rows = []
    for (model, dataset), files in groups.items():
        base_id = f"EXP_D_{model}_{dataset}_TEMPORAL"
        _check_seeds(files, base_id=base_id)
        metrics_per_seed: Dict[str, List[float]] = defaultdict(list)
        for f in files:
            d   = _load(f)
            met = d.get("metrics", {})
            for metric_name in (
                "f1_macro", "accuracy", "mcc",
                "js_divergence_unigrams",
            ):
                v = met.get(metric_name)
                if v is not None:
                    try:
                        metrics_per_seed[metric_name].append(
                            float(v)
                        )
                    except (TypeError, ValueError):
                        pass
        if "js_divergence_unigrams" not in metrics_per_seed:
            jsd_missing_count += 1
        row: Dict[str, Any] = {
            "model":           model,
            "model_display":   _m(model),
            "dataset":         dataset,
            "dataset_display": _ds(dataset),
            "n_seeds":         len(files),
        }
        for metric_name, values in metrics_per_seed.items():
            mean, std = _mean_std(values)
            row[f"{metric_name}_mean"]      = mean
            row[f"{metric_name}_std"]       = std
            row[f"{metric_name}_formatted"] = _fmt(mean, std)
        # Compatibilidade retroativa
        row["f1_macro"] = row.get("f1_macro_mean")
        row["accuracy"] = row.get("accuracy_mean")
        row["mcc"]      = row.get("mcc_mean")
        rows.append(row)
    if jsd_missing_count > 0:
        print(
            f"[AVISO] js_divergence_unigrams ausente em "
            f"{jsd_missing_count} experimento(s) EXP_D. "
            f"Integre calcular_jsd_classes ao pipeline EXP_D "
            f"e regenere os JSONs."
        )
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(
            ["model", "dataset"]
        ).reset_index(drop=True)
        df.to_csv(
            METRICS_DIR / "temporal_summary.csv",
            index=False,
        )
        print(
            f"[OK] temporal_summary.csv  "
            f"({len(df)} linhas)"
        )
    else:
        print("[AVISO] Nenhum EXP_D_* encontrado.")
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  build_summary_csvs — multi-seed (v6)")
    print(f"  Lendo: {METRICS_DIR.resolve()}")
    print(f"  Seeds esperadas: {list(EXPECTED_SEEDS)}")
    print("=" * 62 + "\n")
    cross_df = build_cross_dataset()
    build_generalization_gap(cross_df)
    build_robustness()
    build_combined_summary()
    build_temporal_summary()
    print("\n[PRONTO] Próximos passos:")
    print("  python scripts/generate_tables.py")
    print("  python scripts/generate_figures.py")