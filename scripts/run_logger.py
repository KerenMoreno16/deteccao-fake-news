# -*- coding: utf-8 -*-
"""
scripts/run_logger.py — Logger automático de análises
para a monografia.
Correções aplicadas:
  - cmd_scan() normaliza experiment_id removendo sufixo
    _seed{N} para evitar entradas duplicadas por seed.
  - cmd_log() documenta     explicitamente que métricas manuais sobrescrevem
    métricas auto-carregadas.
  - Rótulos de modelos distinguem BERT (EN) e
    BERTimbau (PT).
  - cmd_report() agrupa entradas por experimento base,
    não por seed.
  - import numpy movido para o topo do arquivo.
  - EXPECTED_SEEDS declarado como tupla imutável.
  - _escape_tex() corrigido: \\ substituído primeiro.
  - auto_metrics.get(k) corrigido com fallback 0.
  - groups em cmd_scan() com tipo correto.
  - Separadores de comentário corrigidos.
"""
import argparse
import json
import re
import sys
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LOG_FILE = ROOT / "results" / "run_log.jsonl"
TEX_OUT  = (
    ROOT
    / "latex"
    / "report_fragments"
    / "run_history.tex"
)
# Declarada como tupla para evitar mutação acidental
EXPECTED_SEEDS: Tuple[int, ...] = tuple(range(42, 52))
DATASET_LABELS: Dict[str, str] = {
    "faketruebr":         "FakeTrue.Br",
    "fakebr":             "Fake.br",
    "fakerecogna":        "FakeRecogna",
    "fakenewsnet":        "FakeNewsNet",
    "constraint_covid19": "CONSTRAINT-COVID19",
    "pheme":              "PHEME",
}
# Rótulos distintos para BERT (EN) e BERTimbau (PT)
MODEL_LABELS: Dict[str, str] = {
    "logistic_regression": r"LR + TF-IDF",
    "mlp":                 r"MLP + TF-IDF",
    "bilstm":              r"BiLSTM",
    "bert":                r"BERT (EN)",
    "bertimbau":           r"BERTimbau (PT)",
}
EXPERIMENT_LABELS: Dict[str, str] = {
    "EXP_A": "Intra-dataset",
    "EXP_B": "Cross-dataset",
    "EXP_C": "Combined PT-BR",
    "EXP_D": "Temporal",
    "EXP_E": "Robustez",
    "EXP_F": "Explicabilidade",
}
# ── Helpers ───────────────────────────────────────────────────────────────────
def _ds(name: str) -> str:
    return DATASET_LABELS.get(name, name)
def _model(name: str) -> str:
    return MODEL_LABELS.get(name, name)
def _exp_type(exp_id: str) -> str:
    clean = re.sub(r"_seed\d+$", "", exp_id)
    for prefix, label in EXPERIMENT_LABELS.items():
        if clean.upper().startswith(prefix):
            return label
    return clean
def _fmt(v) -> str:
    if v is None:
        return "N/D"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)
def _escape_tex(s: str) -> str:
    """
    Escapa caracteres especiais LaTeX.
    A substituição de \\ deve ocorrer PRIMEIRO para evitar
    que as barras introduzidas pelas substituições seguintes
    sejam elas mesmas escapadas.
    """
    for ch, rep in [
        ("\\", r"\textbackslash{}"),   # primeiro
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]:
        s = s.replace(ch, rep)
    return s
def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def _normalize_exp_id(exp_id: str) -> str:
    """
    Remove sufixo _seed{N} do experiment_id para obter
    o ID base. Usado para deduplicação no log.
    """
    return re.sub(r"_seed\d+$", "", exp_id)
def _load_log() -> List[Dict]:
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries
def _append_log(entry: Dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(entry, ensure_ascii=False) + "\n"
        )
    print(
        f"  [OK] Entrada adicionada ao log: {LOG_FILE}"
    )
def _load_result_json(
    exp_id: str,
) -> Optional[Dict]:
    """
    Tenta carregar o JSON correspondente ao experimento.
    Procura primeiro pelo ID exato, depois pelo ID base
    (sem sufixo de seed).
    """
    base_id    = _normalize_exp_id(exp_id)
    candidates = [
        ROOT / "results" / "metrics" / f"{exp_id}.json",
        ROOT / "results" / "metrics" / f"{base_id}.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                return json.loads(
                    c.read_text(encoding="utf-8")
                )
            except Exception:
                pass
    return None
# ── Subcommand: log ───────────────────────────────────────────────────────────
def cmd_log(args) -> None:
    """
    Registra uma execução no log.
    Ordem de prioridade das métricas:
    1. Métricas passadas manualmente via --f1,
       --accuracy, etc.
    2. Métricas auto-carregadas do JSON de resultado.
    ATENÇÃO: métricas manuais sobrescrevem as
    auto-carregadas. Se passar --f1 manualmente,
    certifique-se de que o valor corresponde ao
    resultado oficial registrado no JSON. Divergências
    entre o log e os JSONs podem ser identificadas
    pelo subcomando 'scan'.
    """
    exp_id = _normalize_exp_id(args.experiment or "")
    entry: Dict[str, Any] = {
        "timestamp":  _now(),
        "experiment": exp_id,
        "command":    args.command or "",
        "notes":      args.notes or "",
        "step":       args.step or "",
    }
    # Métricas manuais
    manual_metrics: Dict[str, float] = {}
    if args.f1        is not None:
        manual_metrics["f1_macro"]        = args.f1
    if args.accuracy  is not None:
        manual_metrics["accuracy"]        = args.accuracy
    if args.precision is not None:
        manual_metrics        ["precision_macro"] = args.precision
    if args.recall    is not None:
        manual_metrics["recall_macro"]    = args.recall
    if args.roc_auc   is not None:
        manual_metrics["roc_auc"]         = args.roc_auc
    if args.mcc       is not None:
        manual_metrics["mcc"]             = args.mcc
    # Auto-carregamento do JSON
    auto_metrics: Dict[str, float] = {}
    if args.experiment:
        result = _load_result_json(args.experiment)
        if result:
            auto_metrics = result.get("metrics", {})
            if not entry["step"]:
                entry["step"] = "auto-loaded"
    # Merge: manual tem prioridade sobre auto
    # Documentado explicitamente para rastreabilidade
    merged = {**auto_metrics, **manual_metrics}
    if manual_metrics and auto_metrics:
        overlap = set(manual_metrics) & set(auto_metrics)
        if overlap:
            for k in overlap:
                auto_val = float(
                    auto_metrics.get(k, 0)
                )
                man_val  = float(manual_metrics[k])
                if abs(man_val - auto_val) > 1e-4:
                    print(
                        f"  [AVISO] Métrica '{k}' "
                        f"diverge: "
                        f"manual={man_val:.4f}, "
                        f"JSON={auto_val:.4f}. "
                        f"O valor manual será "
                        f"registrado. "
                        f"Verifique a consistência."
                    )
    entry["metrics"] = merged
    # Executa o comando se solicitado
    if args.command and args.run:
        print(f"\n  Executando: {args.command}\n")
        t0 = datetime.now()
        try:
            result_proc = subprocess.run(
                args.command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(ROOT),
                encoding="utf-8",
                errors="replace",
            )
            elapsed = (
                datetime.now() - t0
            ).total_seconds()
            entry["return_code"] = (
                result_proc.returncode
            )
            entry["stdout_tail"] = (
                result_proc.stdout[-3000:]
            )
            entry["stderr_tail"] = (
                result_proc.stderr[-1000:]
            )
            entry["elapsed_s"] = round(elapsed, 2)
            print(result_proc.stdout[-2000:])
            if result_proc.returncode != 0:
                print(
                    f"\n  [AVISO] Comando retornou "
                    f"código "
                    f"{result_proc.returncode}"
                )
                print(result_proc.stderr[-500:])
        except Exception as e:
            entry["error"] = str(e)
            print(f"  [ERRO] {e}")
    _append_log(entry)
# ── Subcommand: summary ───────────────────────────────────────────────────────
def cmd_summary(args) -> None:
    entries = _load_log()
    if not entries:
        print(
            "  Log vazio. Nenhuma entrada encontrada."
        )
        return
    print(f"\n{'='*72}")
    print(
        f"  HISTÓRICO DE ANÁLISES — "
        f"{len(entries)} entradas"
    )
    print(f"  Arquivo: {LOG_FILE}")
    print(f"{'='*72}")
    print(
        f"  {'#':>3}  {'Timestamp':<20}  "
        f"{'Experimento':<45}  {'F1-macro':>8}"
    )
    print(
        f"  {'-'*3}  {'-'*20}  "
        f"{'-'*45}  {'-'*8}"
    )
    for i, e in enumerate(entries, 1):
        ts  = e.get("timestamp", "")[:19]
        exp = e.get(
            "experiment", e.get("step", "")
        )[:44]
        f1  = e.get("metrics", {}).get("f1_macro")
        f1s = _fmt(f1) if f1 is not None else "     —"
        print(
            f"  {i:>3}  {ts:<20}  "
            f"{exp:<45}  {f1s:>8}"
        )
    print(f"{'='*72}\n")
# ── Subcommand: report ────────────────────────────────────────────────────────
def cmd_report(args) -> None:
    """
    Gera fragmento LaTeX do histórico completo.
    Agrupa entradas por experimento base (sem sufixo
    _seed{N}) para evitar tabelas com uma linha por seed.
    """
    entries = _load_log()
    if not entries:
        print(
            "  Log vazio. Nenhum fragmento gerado."
        )
        return
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    sep = "% " + "=" * 68
    lines = [
        sep,
        "% run_history.tex — Gerado automaticamente "
        "por scripts/run_logger.py",
        "% Inclua no .tex principal com:",
        r"%   \input{latex/report_fragments/run_history}",
        sep,
        r"",
        r"\section*{Histórico de Execuções e Resultados}",
        r"% Este arquivo é gerado automaticamente.",
        r"% Não edite manualmente.",
        r"",
    ]
    # Agrupa por prefixo de experimento
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        exp_id  = _normalize_exp_id(
            e.get("experiment", e.get("step", "misc"))
        )
        matched = False
        for prefix in EXPERIMENT_LABELS:
            if exp_id.upper().startswith(prefix):
                groups[prefix].append(e)
                matched = True
                break
        if not matched:
            groups["MISC"].append(e)
    for prefix in (
        list(EXPERIMENT_LABELS.keys()) + ["MISC"]
    ):
        group_entries = groups.get(prefix, [])
        if not group_entries:
            continue
        label = EXPERIMENT_LABELS.get(prefix, "Outros")
        lines += [
            "",
            f"\\subsection*{{Experimento "
            f"{prefix.replace('EXP_', '')} "
            f"--- {label}}}",
            "",
        ]
        # Agrega métricas por experimento base
        agg_by_exp: Dict[str, Dict] = {}
        for e in group_entries:
            exp_id = _normalize_exp_id(
                e.get("experiment", e.get("step", ""))
            )
            if exp_id not in agg_by_exp:
                agg_by_exp[exp_id] = {
                    "experiment":   exp_id,
                    "notes":        e.get("notes", ""),
                    "timestamp":    e.get(
                        "timestamp", ""
                    ),
                    "command":      e.get("command", ""),
                    "metrics_list": [],
                }
            m = e.get("metrics", {})
            if m:
                agg_by_exp[exp_id][
                    "metrics_list"
                ].append(m)
        # Consolida métricas (média entre seeds)
        consolidated = []
        for exp_id, data in agg_by_exp.            items():
            ml = data["metrics_list"]
            if ml:
                keys  = set(k for d in ml for k in d)
                agg_m: Dict[str, float] = {}
                for k in keys:
                    vals: List[float] = []
                    for d in ml:
                        v = d.get(k)
                        if v is not None:
                            try:
                                vals.append(float(v))
                            except (
                                TypeError,
                                ValueError,
                            ):
                                pass
                    if vals:
                        agg_m[k] = float(
                            np.mean(vals)
                        )
                data["metrics"] = agg_m
            else:
                data["metrics"] = {}
            consolidated.append(data)
        # Tabela consolidada
        metric_entries = [
            e for e in consolidated
            if e.get("metrics")
        ]
        if metric_entries:
            lines += [
                r"\begin{table}[ht]",
                r"\centering",
                r"\small",
                r"\begin{tabular}{llcccc}",
                r"\hline",
                (
                    r"\textbf{Experimento} & "
                    r"\textbf{Notas} & "
                    r"\textbf{F1-macro} & "
                    r"\textbf{Acurácia} & "
                    r"\textbf{ROC-AUC} & "
                    r"\textbf{MCC} \\"
                ),
                r"\hline",
            ]
            for e in metric_entries:
                exp_id = _escape_tex(
                    e.get("experiment", "")[:40]
                )
                notes  = _escape_tex(
                    (e.get("notes") or "")[:40]
                )
                m   = e.get("metrics", {})
                f1  = _fmt(m.get("f1_macro"))
                acc = _fmt(m.get("accuracy"))
                auc = _fmt(m.get("roc_auc"))
                mcc = _fmt(m.get("mcc"))
                lines.append(
                    f"  {exp_id} & {notes} & "
                    f"{f1} & {acc} & "
                    f"{auc} & {mcc} \\\\"
                )
            lines += [
                r"\hline",
                r"\end{tabular}",
                (
                    f"\\caption{{Resultados do "
                    f"Experimento "
                    f"{prefix.replace('EXP_', '')} "
                    f"--- {label} "
                    f"(médias consolidadas sobre "
                    f"seeds registradas). "
                    f"Gerado em "
                    f"\\texttt{{{_escape_tex(_now())}}}."
                    f"}}"
                ),
                f"\\label{{tab:run_log_"
                f"{prefix.lower()}}}",
                r"\end{table}",
                r"",
            ]
        # Parágrafo narrativo
        lines.append(r"\paragraph{Detalhes}")
        for e in consolidated:
            ts     = e.get("timestamp", "")[:16]
            exp_id = e.get("experiment", "")
            cmd    = e.get("command", "")
            notes  = e.get("notes", "")
            m      = e.get("metrics", {})
            n_seeds = len(
                e.get("metrics_list", [])
            )
            parts = [
                f"\\textbf{{{_escape_tex(exp_id)}}}"
            ]
            if notes:
                parts.append(
                    f"({_escape_tex(notes)})"
                )
            if m.get("f1_macro") is not None:
                parts.append(
                    f"F1-macro = "
                    f"\\texttt{{{_fmt(m['f1_macro'])}}}"
                    + (
                        f" ({n_seeds} seeds)"
                        if n_seeds > 1
                        else ""
                    )
                )
            if m.get("accuracy") is not None:
                parts.append(
                    f"Acurácia = "
                    f"\\texttt{{{_fmt(m['accuracy'])}}}"
                )
            if cmd:
                parts.append(
                    f"Comando: "
                    f"\\texttt{{{_escape_tex(cmd[:80])}}}"
                )
            sentence = (
                ", ".join(parts) + f" [{ts}]."
            )
            lines.append(f"  {sentence}")
            lines.append("")
    # Seção de comandos executados
    commands = [
        e for e in entries if e.get("command")
    ]
    if commands:
        lines += [
            r"",
            r"\subsection*{Comandos Executados}",
            r"",
            r"\begin{verbatim}",
        ]
        for e in commands:
            ts     = e.get("timestamp", "")[:19]
            cmd    = e.get("command", "")
            rc     = e.get("return_code")
            rc_str = (
                f"  # exitcode={rc}"
                if rc is not None
                else ""
            )
            lines.append(
                f"[{ts}] {cmd}{rc_str}"
            )
        lines.append(r"\end{verbatim}")
        lines.append(r"")
    # Rodapé
    lines += [
        r"",
        r"\vspace{1em}",
        (
            f"\\noindent\\small{{\\textit{{"
            f"Histórico gerado em {_now()} --- "
            f"{len(entries)} entradas registradas."
            f"}}}}"
        ),
        r"",
    ]
    content = "\n".join(lines) + "\n"
    TEX_OUT.write_text(content, encoding="utf-8")
    n = content.count("\n")
    print(
        f"\n  [OK] Fragmento LaTeX gerado: "
        f"{TEX_OUT}  ({n} linhas)"
    )
    print("  Para incluir no Overleaf:")
    print(
        r"    \input{latex/report_fragments/"
        r"run_history}"
    )
    print()
# ── Subcommand: scan ──────────────────────────────────────────────────────────
def cmd_scan(args) -> None:
    """
    Varre todos os JSONs de resultados e adiciona
    entradas faltantes ao log.
    Normaliza o experiment_id removendo sufixo _seed{N}
    antes de verificar duplicatas. Isso evita registrar
    uma entrada separada para cada seed do mesmo
    experimento.
    O log consolida por experimento base; os valores de
    métricas registrados são os da primeira seed
    encontrada. Para obter médias consolidadas, usar o
    subcomando 'report'.
    """
    metrics_dir = ROOT / "results" / "metrics"
    if not metrics_dir.exists():
        print(
            "  Diretório results/metrics/ "
            "não encontrado."
        )
        return
    # Normaliza IDs existentes no log
    existing: set = {
        _normalize_exp_id(
            e.get("experiment", "")
        )
        for e in _load_log()
    }
    skip_files = {
        "leakage_report.json",
        "data_validation_report.json",
        "environment.json",
    }
    # Agrupa arquivos por experimento base
    # Tipo corrigido: List[Tuple[Path, Dict]]
    groups: Dict[
        str, List[Tuple[Path, Dict]]
    ]     = defaultdict(list)
    for jf in sorted(metrics_dir.glob("EXP_*.json")):
        if jf.name in skip_files:
            continue
        try:
            data = json.loads(
                jf.read_text(encoding="utf-8")
            )
        except Exception:
            continue
        exp_id = _normalize_exp_id(
            data.get("experiment_id", jf.stem)
        )
        groups[exp_id].append((jf, data))
    new_count = 0
    for base_id, file_data_list in groups.items():
        if base_id in existing:
            continue
        # Usa a primeira entrada disponível
        # para metadados
        first_path, first_data = file_data_list[0]
        metrics = first_data.get("metrics", {})
        # Alerta se grupo tem menos seeds que esperado
        if len(file_data_list) < len(EXPECTED_SEEDS):
            print(
                f"  [AVISO] {base_id}: "
                f"apenas {len(file_data_list)} "
                f"seed(s) encontrada(s) no scan "
                f"(esperadas {len(EXPECTED_SEEDS)})."
            )
        entry: Dict[str, Any] = {
            "timestamp":  first_data.get(
                "timestamp", _now()
            ),
            "experiment": base_id,
            "command":    "",
            "notes":      (
                f"auto-scan: "
                f"{len(file_data_list)} "
                f"seed(s) encontrada(s)"
            ),
            "step":          "scan",
            "n_seeds_found": len(file_data_list),
            "metrics": {
                k: metrics.get(k)
                for k in (
                    "f1_macro",
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "roc_auc",
                    "pr_auc",
                    "mcc",
                    "train_time_s",
                    "f1_macro_ci_lo",
                    "f1_macro_ci_hi",
                )
                if metrics.get(k) is not None
            },
        }
        _append_log(entry)
        existing.add(base_id)
        new_count += 1
    if new_count:
        print(
            f"  [OK] {new_count} entradas novas "
            f"adicionadas ao log."
        )
    else:
        print(
            "  Nenhuma entrada nova encontrada — "
            "log já está atualizado."
        )
# ── Subcommand: clear ─────────────────────────────────────────────────────────
def cmd_clear(args) -> None:
    if not args.confirm:
        print(
            "  Use --confirm para limpar o log."
        )
        return
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    print(f"  [OK] Log limpo: {LOG_FILE}")
# ── Subcommand: full ──────────────────────────────────────────────────────────
def cmd_full(args) -> None:
    """Executa scan + report em sequência."""
    print(
        "  [1/2] Escaneando JSONs de resultados..."
    )
    cmd_scan(args)
    print("  [2/2] Gerando fragmento LaTeX...")
    cmd_report(args)
# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Logger automático de análises "
            "para a monografia"
        ),
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
        epilog=textwrap.dedent("""
        Exemplos:
          # Escanear resultados e gerar LaTeX:
          python scripts/run_logger.py full
          # Registrar resultado manual:
          python scripts/run_logger.py log \\
              --experiment EXP_A_bertimbau_faketruebr \\
              --f1 0.9312 --accuracy 0.9315 \\
              --notes "BERTimbau fine-tuning 5 epocas"
          # Executar e registrar automaticamente:
          python scripts/run_logger.py log \\
              --experiment EXP_A_lr_fakebr \\
              --command "python train_models.py \\
                  --dataset fakebr" \\
              --run
          # Ver histórico no terminal:
          python scripts/run_logger.py summary
          # Gerar fragmento LaTeX:
          python scripts/run_logger.py report
          # Limpar log:
          python scripts/run_logger.py clear --confirm
        """),
    )
    sub = ap.add_subparsers(
        dest="subcommand", required=True
    )
    # ── log ──────────────────────────────────────────
    p_log = sub.add_parser(
        "log",
        help="Registrar uma execução de análise",
    )
    p_log.add_argument(
        "--experiment",
        default="",
        help=(
            "ID do experimento base "
            "(ex: EXP_A_lr_fakebr). "
            "Sufixos _seed{N} são removidos "
            "automaticamente."
        ),
    )
    p_log.add_argument(
        "--command",
        default="",
        help="Comando executado",
    )
    p_log.add_argument(
        "--notes",
        default="",
        help="Anotação livre",
    )
    p_log.add_argument(
        "--step",
        default="",
        help=(
            "Etapa do pipeline (ex: step_4)"
        ),
    )
    p_log.add_argument(
        "--run",
        action="store_true",
        help=(
            "Executar o --command e "
            "capturar saída no log"
        ),
    )
    p_log.add_argument(
        "--f1",
        type=float,
        default=None,
        help=(
            "F1-macro. ATENÇÃO: sobrescreve o "
            "valor do JSON de resultado se diferente."
        ),
    )
    p_log.add_argument(
        "--accuracy",
        type=float,
        default=None,
        help="Acurácia",
    )
    p_log.add_argument(
        "--precision",
        type=float,
        default=None,
        help="Precisão macro",
    )
    p_log.add_argument(
        "--recall",
        type=float,
        default=None,
        help="Recall macro",
    )
    p_log.add_argument(
        "--roc-auc",
        dest="roc_auc",
        type=float,
        default=None,
        help="ROC-AUC",
    )
    p_log.add_argument(
        "--mcc",
        type=float,
        default=None,
        help="MCC",
    )
    p_log.set_defaults(func=cmd_log)
    # ── summary ──────────────────────────────────────
    p_sum = sub.add_parser(
        "summary",
        help=(
            "Mostrar resumo do histórico "
            "no terminal"
        ),
    )
    p_sum.set_defaults(func=cmd_summary)
    # ── report ───────────────────────────────────────
    p_rep = sub.add_parser(
        "report",
        help=(
            "Gerar fragmento La            TeX do histórico completo"
        ),
    )
    p_rep.set_defaults(func=cmd_report)
    # ── scan ─────────────────────────────────────────
    p_scan = sub.add_parser(
        "scan",
        help=(
            "Escanear JSONs de resultados "
            "e adicionar ao log"
        ),
    )
    p_scan.set_defaults(func=cmd_scan)
    # ── full ─────────────────────────────────────────
    p_full = sub.add_parser(
        "full",
        help="scan + report em um único comando",
    )
    p_full.set_defaults(func=cmd_full)
    # ── clear ─────────────────────────────────────────
    p_clr = sub.add_parser(
        "clear",
        help="Limpar o log (irreversível)",
    )
    p_clr.add_argument(
        "--confirm",
        action="store_true",
        help="Confirmar limpeza",
    )
    p_clr.set_defaults(func=cmd_clear)
    args = ap.parse_args()
    args.func(args)
if __name__ == "__main__":
    main()