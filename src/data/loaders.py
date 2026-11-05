"""
loaders.py — Dataset loaders para os corpora utilizados no trabalho.

Datasets ativos:
  - FakeTrue.Br        (faketruebr)
  - Fake.br-Corpus     (fakebr)
  - FakeRecogna        (fakerecogna)
  - FakeNewsNet        (fakenewsnet)
  - CONSTRAINT-COVID19 (constraint_covid19)
  - PHEME              (pheme)
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from src.utils.logging import get_logger

logger = get_logger(__name__)

# NOTA: estratégia padrão alterada de "as_fake" para "discard" (Issue #7).
# Com "as_fake", ~95 % dos rótulos fake eram rumores não verificados,
# introduzindo label noise severo. Com "discard" sobram ≈4.105 registros
# (≈4.005 real + ≈100 false), o que deve ser reportado como limitação.
_PHEME_DEFAULT_STRATEGY = "discard"


def _safe_label_map(
    series: pd.Series,
    label_map: dict,
    dataset_name: str = "",
) -> pd.Series:
    """
    Mapeia labels usando label_map.
    Retorna NaN para valores não mapeados.
    """
    mapped = series.map(label_map)
    unknown = mapped.isna() & series.notna()
    if unknown.any():
        n = int(unknown.sum())
        bad_vals = series[unknown].value_counts().to_dict()
        logger.warning(
            "[%s] %d label(s) não mapeado(s) serão descartados: %s",
            dataset_name, n, bad_vals,
        )
    return mapped


def _apply_label_map(
    series: pd.Series,
    label_map: dict,
) -> pd.Series:
    """
    Map raw label strings to integers using label_map.
    Retorna pd.Series com dtype int64 (-1 para não mapeados).
    """
    mapped = series.map(label_map)
    unknown = mapped.isna()
    if unknown.any():
        n_unknown = int(unknown.sum())
        unknown_values = series[unknown].value_counts().to_dict()
        logger.warning(
            "Found %d rows with unknown labels: %s",
            n_unknown, unknown_values,
        )
    return mapped.fillna(-1).astype("int64")


def _verify_fakerecogna_label_convention(
    df: pd.DataFrame,
    label_col: str,
    title_col: Optional[str] = None,
) -> bool:
    """
    Verifica a convenção numérica do FakeRecogna.

    Convenção ORIGINAL do corpus (verificada em inspeção amostral): Classe 0 = fake,
    Classe 1 = real (títulos da classe 0 contêm '#boato'). O pipeline usa 0=real/1=fake.

    Retorna True se os rótulos são numéricos {0, 1} e a convenção original foi
    confirmada (taxa de 'boato' no título maior na classe 0). Levanta ValueError se
    a evidência indicar o contrário (convenção já invertida ou corpus diferente).
    """
    num = pd.to_numeric(df[label_col], errors="coerce")
    vals = set(num.dropna().astype(int).unique().tolist())
    if vals != {0, 1}:
        return False
    if title_col is None or title_col not in df.columns:
        logger.warning("FakeRecogna: sem coluna de título para confirmar a convenção.")
        return True
    has_boato = df[title_col].astype(str).str.contains("boato", case=False, na=False)
    r0 = float(has_boato[num == 0].mean())
    r1 = float(has_boato[num == 1].mean())
    logger.info(
        "FakeRecogna: taxa de 'boato' no título — Classe 0: %.1f%% | Classe 1: %.1f%%",
        100 * r0, 100 * r1,
    )
    if r0 <= r1:
        raise ValueError(
            "FakeRecogna: a convenção esperada (Classe 0 = fake) NÃO foi confirmada "
            f"(boato: classe0={r0:.3f} <= classe1={r1:.3f}). Revise os rótulos."
        )
    return True


def _read_csv_safe(fpath: Path, logger_=None) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(fpath, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(fpath, encoding="utf-8", encoding_errors="replace")


def _map_fakebr_label(s: pd.Series) -> pd.Series:
    """Fake.br (HF parquet): label bool, False=fake, True=real."""
    if s.dtype == bool:
        return (~s).astype("int64")          # False(fake)->1 ; True(real)->0
    lm = {"fake": 1, "false": 1, "f": 1, "true": 0, "real": 0, "t": 0,
          "1": 1, "0": 0}                    # CSV GitHub já vem fake=1/true=0
    out = s.astype(str).str.strip().str.lower().map(lm)
    return out


def load_faketruebr(data_dir: str) -> pd.DataFrame:
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("*.parquet"))
    label_map = {
        "fake": 1, "true": 0, "real": 0,
        "1": 1, "0": 0, 1: 1, 0: 0,
    }
    if parquet_files:
        frames = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(frames, ignore_index=True)
        logger.info(
                "FakeTrue.Br | Parquet encontrados: %s",
                [str(p) for p in parquet_files],
            )
        if "text" in df.columns and "label" in df.columns:
            df["label"] = _safe_label_map(
                df["label"].astype(str).str.strip().str.lower(),
                label_map,
                dataset_name="faketruebr",
            )
            df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
            df["label"] = df["label"].astype(int)
            df["dataset"] = "faketruebr"
            return df[["text", "label", "dataset"]]
    
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files and not parquet_files:
        raise FileNotFoundError(f"No FakeTrue.Br data found in {data_dir}.")
    raw = pd.concat([_read_csv_safe(p, logger) for p in csv_files], ignore_index=True)
    logger.info(
        "FakeTrue.Br | CSV encontrados: %s",
        [str(p) for p in csv_files],
    )
    logger.info(
        "[FakeTrue.Br] CSV lido: %d linhas | colunas=%s",
        len(raw),
        raw.columns.tolist(),
    )
    if "fake" in raw.columns and "true" in raw.columns and "label" not in raw.columns:
        fake_text_col = next((c for c in ["fake", "fake_text", "texto_fake"] if c in raw.columns), "fake")
        true_text_col = next((c for c in ["true", "real", "true_text", "texto_true"] if c in raw.columns), "true")
        records = []
        # Simetria entre classes: o CSV só traz título para a classe fake
        # ('title_fake'); concatená-lo criaria um atalho estrutural (Geirhos et al., 2020).
        # Por isso o título NÃO é usado: ambas as classes usam apenas o corpo do texto.
        for _, row in raw.iterrows():
            fake_text = str(row[fake_text_col]).strip()
            true_text = str(row[true_text_col]).strip()
            if fake_text and fake_text.lower() not in ("nan", "none"):
                records.append({"text": fake_text, "label": 1})
            if true_text and true_text.lower() not in ("nan", "none"):
                records.append({"text": true_text, "label": 0})
        df = pd.DataFrame(records)
        logger.info(
            "[FakeTrue.Br] Após expansão: %d registros | fake=%d | real=%d",
            len(records),
            sum(r["label"] == 1 for r in records),
            sum(r["label"] == 0 for r in records),
        )
    else:
        text_col = next((c for c in ["text", "preprocessed_news", "news", "content", "noticia"] if c in raw.columns), None)
        label_col = next((c for c in ["label", "class", "target", "rotulo"] if c in raw.columns), None)
        if text_col is None or label_col is None:
            raise ValueError(f"Cannot identify text/label columns in FakeTrue.Br. Columns: {raw.columns.tolist()}")
        df = raw.copy()
        df["text"] = df[text_col].astype(str)
        df["label"] = _safe_label_map(
            df[label_col].astype(str).str.strip().str.lower(),
            label_map,
            dataset_name="faketruebr",
        )
    logger.info(
        "[FakeTrue.Br] Antes do dropna: %d | fake=%d | real=%d",
        len(df),
        int((df["label"] == 1).sum()),
        int((df["label"] == 0).sum()),
    )
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["dataset"] = "faketruebr"
    return df[["text", "label", "dataset"]]


def load_fakebr(data_dir: str) -> pd.DataFrame:
    data_path = Path(data_dir)

    parquet_files = sorted(data_path.glob("*.parquet"))
    if parquet_files:
        df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    elif list(data_path.glob("*.csv")):
        df = pd.concat([_read_csv_safe(p, logger) for p in data_path.glob("*.csv")], ignore_index=True)
    else:
        label_dirs_candidates = [("fake", 1), ("true", 0), ("full_fake", 1), ("full_true", 0)]
        found_dirs = {}
        for dir_name, label_int in label_dirs_candidates:
            candidate = data_path / dir_name
            if candidate.exists():
                base = "fake" if label_int == 1 else "true"
                if base not in found_dirs:
                    found_dirs[base] = (candidate, label_int)
        if not found_dirs:
            raise FileNotFoundError(f"Fake.br: nenhum diretório de labels encontrado em {data_path}.")
        records = []
        for base_label, (label_dir, label_int) in found_dirs.items():
            for txt_file in sorted(label_dir.glob("*.txt")):
                try:
                    text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
                    if text:
                        records.append({"text": text, "label": label_int})
                except OSError:
                    continue
        df = pd.DataFrame(records)

    text_col = next((c for c in ["preprocessed_news", "text", "news", "content"] if c in df.columns), None)
    label_col = next((c for c in ["label", "class", "target"] if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Cannot identify text/label columns in Fake.br. Columns: {df.columns.tolist()}")

    df["text"] = df[text_col].astype(str)
    # Parquet HF: label bool (False=fake, True=real). CSV GitHub: fake=1/true=0.
    # Qualquer rótulo não mapeado interrompe o carregamento (sem descarte silencioso).
    df["label"] = _map_fakebr_label(df[label_col])
    n_bad = int(df["label"].isna().sum())
    if n_bad:
        raise ValueError(
            f"Fake.br: {n_bad} rótulo(s) não mapeado(s) em "
            f"{df.loc[df['label'].isna(), label_col].value_counts().to_dict()}"
        )
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df["label"] = df["label"].astype("int64")
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(
            f"Fake.br deve ter 2 classes; encontrado {df['label'].value_counts().to_dict()}"
        )
    logger.info(
        "Fake.br: %d registros, distribuição %s (label bool: False=fake→1, True=real→0)",
        len(df), df["label"].value_counts().to_dict(),
    )
    df["dataset"] = "fakebr"
    out_cols = ["text", "label", "dataset"]
    if "timestamp" in df.columns:
        out_cols.append("timestamp")
    return df[out_cols]


def _parse_mixed_dates(s: pd.Series) -> pd.Series:
    """
    Parse de datas heterogêneas do FakeRecogna: '11/01/2021' e '27/03/2020 18h25 '.
    Normaliza 'HHhMM' → 'HH:MM' e usa formato misto dia-primeiro.
    """
    clean = (
        s.astype(str).str.strip()
        .str.replace(r"(\d{1,2})h(\d{2})", r"\1:\2", regex=True)
    )
    return pd.to_datetime(clean, format="mixed", dayfirst=True, errors="coerce")


def load_fakerecogna(data_dir: str) -> pd.DataFrame:
    data_path = Path(data_dir)
    df = None
    xlsx_files = sorted(data_path.glob("*.xlsx"))
    if xlsx_files:
        try:
            frames = [pd.read_excel(f, engine="openpyxl") for f in xlsx_files]
            df = pd.concat(frames, ignore_index=True)
        except ImportError:
            pass

    if df is None:
        parquet_files = sorted(data_path.glob("*.parquet"))
        if parquet_files:
            df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)

    if df is None:
        csv_files = sorted(data_path.glob("*.csv"))
        if csv_files:
            df = pd.concat([_read_csv_safe(p, logger) for p in csv_files], ignore_index=True)

    if df is None:
        jsonl_files = sorted(list(data_path.glob("*.jsonl")) + list(data_path.glob("*.json")))
        if jsonl_files:
            records = []
            for jf in jsonl_files:
                try:
                    with open(jf, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                records.append(json.loads(line.strip()))
                except Exception:
                    pass
            if records:
                df = pd.DataFrame(records)

    if df is None or df.empty:
        raise FileNotFoundError(f"No FakeRecogna data found in {data_dir}.")

    text_col = next(
        (
            c for c in [
                "Noticia", "text", "news", "content",
                "noticia", "manchete", "Titulo",
                "titulo", "texto",
            ]
            if c in df.columns
        ),
        None,
    )
    label_col = next(
        (
            c for c in [
                "Classe", "classe", "label", "class", "target", "rotulo"
            ] 
            if c in df.columns
        ),
        None
    )
    if text_col is None or label_col is None:
        raise ValueError(f"Cannot identify text/label columns in FakeRecogna. Columns: {df.columns.tolist()}")

    title_col = next((c for c in ["Titulo", "titulo", "manchete"] if c in df.columns), None)
    numeric_original = _verify_fakerecogna_label_convention(df, label_col, title_col)
    df["text"] = df[text_col].astype(str)
    raw_labels = df[label_col]
    if numeric_original:
        # Convenção original do FakeRecogna: 0=fake, 1=real  →  pipeline: 1=fake, 0=real.
        cls = pd.to_numeric(raw_labels, errors="coerce")
        mapped = (1 - cls).where(cls.isin([0, 1]))
        logger.info(
            "FakeRecogna: Classe original {0=fake, 1=real} -> pipeline {1=fake, 0=real} "
            "(%d nulos/inválidos descartados)", int(mapped.isna().sum()),
        )
    else:
        label_map = {
            "fake": 1, "falso": 1, "falsa": 1, "mentira": 1, "boato": 1,
            "true": 0, "real": 0, "verdadeiro": 0, "verdadeira": 0, "fato": 0,
        }
        mapped = _safe_label_map(
            raw_labels.astype(str).str.strip().str.lower(),
            label_map,
            dataset_name="fakerecogna",
        )

    df["label"] = mapped
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["dataset"] = "fakerecogna"

    date_col = next((c for c in ["Data", "data", "date", "Date", "published_at", "published"] if c in df.columns), None)
    if date_col is not None:
        ts = _parse_mixed_dates(df[date_col])
        if int(ts.notna().sum()) > 0:
            df["timestamp"] = ts
            for lbl in (0, 1):
                m = df["label"] == lbl
                logger.info(
                    "FakeRecogna: timestamps válidos classe %d: %d/%d (%.1f%%)",
                    lbl, int(ts[m].notna().sum()), int(m.sum()),
                    100 * float(ts[m].notna().mean()) if m.any() else 0.0,
                )
            # LIMITAÇÃO: cobertura temporal assimétrica entre classes.
            # Notícias falsas abrangem 2012–2021; notícias reais apenas 2019–2021.
            # Isso significa que um modelo temporal pode aprender o período de
            # publicação em vez do conteúdo — registrar como limitação na monografia.
            logger.warning(
                "FakeRecogna: LIMITAÇÃO TEMPORAL — fakes cobrem ~2012-2021, "
                "reais apenas ~2019-2021. O split temporal pode confundir período "
                "de publicação com veracidade. Registrar como limitação na monografia."
            )

    out_cols = ["text", "label", "dataset"]
    if "timestamp" in df.columns:
        out_cols.append("timestamp")
    return df[out_cols]


def load_fakenewsnet(data_dir: str) -> pd.DataFrame:
    records = []
    n_skipped = 0
    for label_str, label_int in [("fake", 1), ("real", 0)]:
        label_dir = Path(data_dir) / label_str
        if not label_dir.exists():
            continue
        for news_dir in sorted(label_dir.iterdir()):
            if not news_dir.is_dir():
                continue
            content_file = news_dir / "news content.json"
            if not content_file.exists():
                continue
            try:
                content = json.loads(content_file.read_text(encoding="utf-8"))
                title_clean = str(content.get("title", "")).strip()
                text_clean = str(content.get("text", "")).strip()
                is_fallback = bool(content.get("text_is_title_fallback", False))
                if text_clean and text_clean != title_clean and not is_fallback:
                    combined = f"{title_clean} {text_clean}".strip()
                elif title_clean:
                    combined = title_clean
                else:
                    combined = text_clean
                if not combined:
                    continue
                record = {
                    "text": combined,
                    "label": label_int,
                    "dataset": "fakenewsnet",
                }
                pub_date = content.get("publish_date", "")
                if isinstance(pub_date, str) and pub_date.strip():
                    ts = pd.to_datetime(pub_date.strip(), errors="coerce")
                    if pd.notna(ts):
                        record["timestamp"] = ts
                records.append(record)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                n_skipped += 1
                logger.warning("FakeNewsNet: arquivo ignorado (%s): %s", content_file, exc)
                continue

    if not records:
        raise FileNotFoundError(f"No news content JSON files found in {data_dir}.")
    if n_skipped:
        logger.warning("FakeNewsNet: %d arquivo(s) ignorado(s) por erro de leitura.", n_skipped)
    df = pd.DataFrame(records).dropna(subset=["text", "label"]).reset_index(drop=True)

    # Mesma notícia presente em fake e real → rótulo ambíguo: remove ambas as cópias.
    key = df["text"].str.strip().str.lower()
    n_labels = df.groupby(key)["label"].transform("nunique")
    n_conflict = int((n_labels > 1).sum())
    if n_conflict:
        logger.warning("FakeNewsNet: %d registro(s) com texto idêntico e rótulos conflitantes removidos.", n_conflict)
        df = df[n_labels == 1].reset_index(drop=True)
    logger.warning(
        "FakeNewsNet: n=%d — BAIXO PODER ESTATÍSTICO no split 70/10/20 (teste ≈ %d). "
        "Interpretar IC bootstrap com cautela (ver metodologia).",
        len(df), int(0.2 * len(df)),
    )
    # O corpus em disco (1052 artigos) não possui datas de publicação —
    # split temporal não é aplicável a este dataset.
    if "timestamp" in df.columns:
        n_ts = int(df["timestamp"].notna().sum())
        if n_ts == 0:
            df = df.drop(columns=["timestamp"])
            logger.warning(
                "FakeNewsNet: nenhuma data de publicação encontrada nos %d artigos. "
                "Split temporal não aplicável — registrar como limitação.",
                len(df),
            )
        else:
            logger.info("FakeNewsNet: %d/%d registros com data de publicação.", n_ts, len(df))
    else:
        logger.warning(
            "FakeNewsNet: campo 'publish_date' ausente no corpus em disco. "
            "Split temporal não aplicável — registrar como limitação.",
        )
    out_cols = ["text", "label", "dataset"]
    if "timestamp" in df.columns:
        out_cols.append("timestamp")
    return df[out_cols]


def load_constraint_covid19(data_dir: str) -> pd.DataFrame:
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("*.parquet"))
    if parquet_files:
        df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    else:
        frames = []
        for fname in ("train.csv", "val.csv", "test.csv"):
            fpath = data_path / fname
            if not fpath.exists():
                continue
            frame = _read_csv_safe(fpath, logger)
            label_col = next((c for c in ["label", "class", "target"] if c in frame.columns), None)
            if label_col is None:
                continue
            if int(frame[label_col].notna().sum()) == 0:
                continue
            frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"No CSV/parquet files found in {data_dir}.")
        df = pd.concat(frames, ignore_index=True)

    text_col = next((c for c in ["tweet", "text", "content"] if c in df.columns), None)
    label_col = next((c for c in ["label", "class", "target"] if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Cannot identify text/label columns in CONSTRAINT-COVID19.")

    label_map = {"fake": 1, "real": 0, "true": 0, "false": 1}
    df["text"] = df[text_col].astype(str)
    df["label"] = _apply_label_map(df[label_col].astype(str).str.strip().str.lower(), label_map)
    df = df[df["label"] != -1].dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["dataset"] = "constraint_covid19"
    return df[["text", "label", "dataset"]]


def load_pheme(data_dir: str, unverified_strategy: str = _PHEME_DEFAULT_STRATEGY) -> pd.DataFrame:
    valid_strategies = {"discard", "as_fake", "as_real"}
    if unverified_strategy not in valid_strategies:
        raise ValueError(f"unverified_strategy inválido: '{unverified_strategy}'. Use: {valid_strategies}")

    records = []
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"PHEME não encontrado em {data_dir}")

    for event_dir in sorted(data_path.iterdir()):
        if not event_dir.is_dir():
            continue
        for rtype in ("rumours", "non-rumours"):
            rdir = event_dir / rtype
            if not rdir.exists():
                continue
            for thread_dir in sorted(rdir.iterdir()):
                if not thread_dir.is_dir():
                    continue
                if rtype == "non-rumours":
                    label_int = 0
                    veracity = "non-rumour"
                else:
                    veracity = "unverified"
                    ann_file = thread_dir / "annotation.json"
                    if ann_file.exists():
                        try:
                            ann = json.loads(ann_file.read_text(encoding="utf-8"))
                            veracity = ann.get("veracity", "unverified").lower()
                        except Exception:
                            veracity = "unverified"
                    if veracity == "false":
                        label_int = 1
                    elif veracity == "true":
                        label_int = 0
                    elif veracity == "unverified":
                        if unverified_strategy == "discard":
                            continue
                        elif unverified_strategy == "as_fake":
                            label_int = 1
                        else:
                            label_int = 0
                    else:
                        continue

                src_dir = thread_dir / "source-tweets"
                if not src_dir.exists():
                    src_dir = thread_dir / "source-tweet"
                if not src_dir.exists():
                    continue
                for tf in sorted(src_dir.glob("*.json")):
                    try:
                        tweet = json.loads(tf.read_text(encoding="utf-8"))
                        text = tweet.get("text", "").strip()
                        if not text:
                            continue
                        record = {
                            "text": text,
                            "label": label_int,
                            "dataset": "pheme",
                            "event": event_dir.name,
                            "veracity": veracity,
                        }
                        created_at = tweet.get("created_at")
                        if created_at:
                            record["timestamp"] = pd.to_datetime(
                                created_at,
                                format="%a %b %d %H:%M:%S +0000 %Y",
                                errors="coerce",
                            )
                        records.append(record)
                        break
                    except Exception:
                        continue

    if not records:
        raise FileNotFoundError(f"Nenhum dado PHEME em {data_dir}")

    df = pd.DataFrame(records).dropna(subset=["text", "label"]).reset_index(drop=True)

    # Verificar desequilíbrio após construção do DataFrame (#A1).
    # Com unverified_strategy="discard", apenas rumores com veredicto explícito
    # são mantidos, resultando em ~100 fakes vs ~4005 reais (ratio ≈ 40:1).
    # Isso deve ser documentado como limitação severa na monografia.
    vcounts = df["label"].value_counts()
    if len(vcounts) >= 2:
        ratio = vcounts.max() / max(vcounts.min(), 1)
        logger.info(
            "PHEME (%s): fake=%d, real=%d (ratio=%.1f:1).",
            unverified_strategy,
            int(vcounts.get(1, 0)), int(vcounts.get(0, 0)), ratio,
        )
        if ratio > 5:
            logger.warning(
                "PHEME (%s): desequilíbrio severo — fake=%d, real=%d (ratio=%.0f:1). "
                "Bootstrap CI pode retornar NaN (n_fake_teste < 30). "
                "Considerar PR-AUC como métrica principal. "
                "Registrar como limitação na monografia.",
                unverified_strategy,
                int(vcounts.get(1, 0)), int(vcounts.get(0, 0)), ratio,
            )

    cols = ["text", "label", "dataset"]
    if "timestamp" in df.columns:
        cols.append("timestamp")
    if "event" in df.columns:
        cols.append("event")
    if "veracity" in df.columns:
        cols.append("veracity")
    return df[cols]

LOADERS = {
    "faketruebr": load_faketruebr,
    "fakebr": load_fakebr,
    "fakerecogna": load_fakerecogna,
    "fakenewsnet": load_fakenewsnet,
    "constraint_covid19": load_constraint_covid19,
    "pheme": load_pheme,
}

LOADER_ALIASES = {
    "faketrue.br": "faketruebr",
    "faketrue_br": "faketruebr",
    "fake.br": "fakebr",
    "fake.br-corpus": "fakebr",
    "fakenewsnet-politifact": "fakenewsnet",
    "fakenewsnet_politifact": "fakenewsnet",
}


def load_dataset(
    name: str,
    data_dir: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    name_lower = name.lower()
    key = LOADER_ALIASES.get(name_lower, name_lower)
    if key not in LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {sorted(LOADERS.keys())}"
        )
    if key == "pheme":
        if cfg is None:
            return load_pheme(data_dir, unverified_strategy=_PHEME_DEFAULT_STRATEGY)
        strategy = cfg.get("datasets", {}).get("pheme", {}).get("unverified_strategy", _PHEME_DEFAULT_STRATEGY)
        return load_pheme(data_dir, unverified_strategy=strategy)
    return LOADERS[key](data_dir)