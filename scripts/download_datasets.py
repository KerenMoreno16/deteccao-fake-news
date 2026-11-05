# -*- coding: utf-8 -*-
"""
scripts/download_datasets.py
Downloader oficial dos 6 datasets utilizados na monografia:
    D-A / faketruebr        FakeTrue.Br
    D-B / fakenewsnet       FakeNewsNet — subconjunto PolitiFact
    D-C / pheme             PHEME
    D-D / fakebr            Fake.br-Corpus
    D-E / fakerecogna       FakeRecogna
    D-F / constraint_covid19  CONSTRAINT COVID-19 Fake News
Uso:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --only da
    python scripts/download_datasets.py --only da de
    python scripts/download_datasets.py --data-dir data/raw
    python scripts/download_datasets.py --allow-demo
IMPORTANTE
----------
Por padrão, falhas de download NÃO geram dados sintéticos e NÃO são
reportadas como sucesso. O flag --allow-demo é exclusivo para testes
locais e NÃO deve ser usado nos experimentos da monografia.
Labels numéricos
----------------
Os datasets são salvos com labels numéricos padronizados:
    1 = notícia falsa (fake)
    0 = notícia verdadeira (real/true)
Esta conversão é aplicada apenas quando o downloader gera o CSV
combinado (Fake.br). Para datasets que já possuem estrutura de
diretórios (FakeNewsNet, PHEME), a conversão é responsabilidade
do loader do pipeline.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Optional
# ── Raiz do projeto ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    import io as _io_fix
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = _io_fix.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = _io_fix.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
try:
    import requests
except ImportError:
    print("[ERROR] requests não está instalado.")
    print("        Instale com: pip install requests")
    sys.exit(1)
# ── Constantes ────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = ROOT / "data" / "raw"
TIMEOUT_DEFAULT  = 60
TIMEOUT_LARGE    = 300
USER_AGENT = (
    "fake-news-study-dataset-downloader/1.0 "
    "(academic reproducibility)"
)
# Labels numéricos padronizados usados pelo pipeline
# 1 = fake / 0 = real (verdadeira)
LABEL_MAP_FAKEBR: Dict[str, int] = {"fake": 1, "true": 0}
DATASET_SPECS: Dict[str, Dict[str, str]] = {
    "da": {
        "name":      "FakeTrueBR",
        "canonical": "faketruebr",
        "directory": "faketruebr",
    },
    "db": {
        "name":      "FakeNewsNet / PolitiFact",
        "canonical": "fakenewsnet",
        "directory": "fakenewsnet",
    },
    "dc": {
        "name":      "PHEME",
        "canonical": "pheme",
        "directory": "pheme",
    },
    "dd": {
        "name":      "Fake.br-Corpus",
        "canonical": "fakebr",
        "directory": "fakebr",
    },
    "de": {
        "name":      "FakeRecogna",
        "canonical": "fakerecogna",
        "directory": "fakerecogna",
    },
    "df": {
        "name":      "CONSTRAINT COVID-19",
        "canonical": "constraint_covid19",
        "directory": "constraint_covid19",
    },
}
# ── Dependências opcionais ────────────────────────────────────────────────────
def _try_import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None
# ── Utilidades ────────────────────────────────────────────────────────────────
def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
def _remove_if_empty(path: Path) -> None:
    try:
        if (
            path.exists()
            and path.is_dir()
            and not any(path.iterdir())
        ):
            path.rmdir()
    except OSError:
        pass
def _request(
    url: str,
    *,
    timeout: int = TIMEOUT_DEFAULT,
    stream: bool = False,
) -> requests.Response:
    """GET com User-Agent e raise_for_status."""
    headers  = {"User-Agent": USER_AGENT}
    response = requests.get(
        url,
        headers=headers,
        timeout=timeout,
        stream=stream,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response
def _download_file(
    url: str,
    destination: Path,
    *,
    timeout: int = TIMEOUT_DEFAULT,
    min_bytes: int = 1,
) -> bool:
    """
    Download atômico via arquivo .part.
    Só move para o destino final após download completo e validação
    do tamanho mínimo.
    """
    _ensure_dir(destination.parent)
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        print(f"      Download: {url}")
        with _request(url, timeout=timeout, stream=True) as response:
            total = 0
            with open(tmp, "wb") as fh:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    total += len(chunk)
        if total < min_bytes:
            raise RuntimeError(
                f"arquivo muito pequeno ({total} bytes); "
                f"mínimo esperado: {min_bytes} bytes."
            )
        os.replace(tmp, destination)
        print(
            f"      [OK] {destination.name} "
            f"({total:,} bytes)"
        )
        return True
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        print(f"      [ERRO] {exc}")
        return False
def _directory_has_files(
    path: Path,
    *,
    recursive: bool = True,
) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    iterator = path.rglob("*") if recursive else path.iterdir()
    return any(p.is_file() for p in iterator)
def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())
def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
def _print_dataset_map() -> None:
    print("\nMapeamento dos datasets:")
    for flag, spec in DATASET_SPECS.items():
        print(
            f"  --only {flag:<2} -> "
            f"{spec['canonical']:<20} -> "
            f"{spec['directory']}/"
        )
    print()
def _normalize_labels(
    df: "pd.DataFrame",
    label_col: str,
    label_map: Dict[str, int],
    dataset_name: str,
) -> "pd.DataFrame":
    """
    Converte labels textuais para inteiros numéricos padronizados.
    Parâmetros
    ----------
    df           : DataFrame pandas.
    label_col    : nome da coluna de labels.
    label_map    : dicionário de mapeamento, ex: {"fake": 1, "true": 0}.
    dataset_name : nome do dataset para mensagens de aviso.
    Retorna
    -------
    DataFrame com a coluna label_col convertida.
    ATENÇÃO
    -------
    Esta função deve ser chamada SOMENTE se confirmado que o loader
    do pipeline espera labels numéricos. Se o loader faz a conversão
    internamente, não chamar esta função para evitar dupla transformação.
    """
    if label_col not in df.columns:
        print(
            f"[AVISO] {dataset_name}: coluna '{label_col}' "
            f"não encontrada — labels não convertidos."
        )
        return df
    unmapped = (
        df[~df[label_col].isin(label_map.keys())][label_col]
        .unique()
    )
    if len(unmapped) > 0:
        print(
            f"[AVISO] {dataset_name}: labels não reconhecidos: "
            f"{unmapped}. Verifique o mapeamento em LABEL_MAP_FAKEBR."
        )
    df = df.copy()
    df[label_col] = df[label_col].map(label_map)
    return df
# ── D-A — FakeTrue.Br ────────────────────────────────────────────────────────
def download_faketruebr(
    out_dir: Path,
    *,
    allow_demo: bool = False,
) -> bool:
    """
    FakeTrue.Br.
    Fonte principal:
        Hugging Face: vzani/corpus-faketrue-br
    Saída:
        data/raw/faketruebr/
    """
    target   = out_dir / "faketruebr"
    _ensure_dir(target)
    existing = (
        list(target.glob("*.csv"))
        + list(target.glob("*.parquet"))
    )
    if existing:
        print(
            f"[D-A] FakeTrueBR já presente em {target}. "
            f"Encontrados {len(existing)} arquivo(s)."
        )
        return True
    print("[D-A] Baixando FakeTrue.Br...")
    candidates = [
        (
            "https://huggingface.co/datasets/"
            "vzani/corpus-faketrue-br/resolve/main/"
            "FakeTrueBr_corpus.csv?download=true",
            target / "FakeTrueBr_corpus.csv",
        ),
        (
            "https://huggingface.co/datasets/"
            "vzani/corpus-faketrue-br/resolve/main/"
            "corpus_train_df.parquet?download=true",
            target / "corpus_train_df.parquet",
        ),
        (
            "https://huggingface.co/datasets/"
            "vzani/corpus-faketrue-br/resolve/main/"
            "corpus_test_df.parquet?download=true",
            target / "corpus_test_df.parquet",
        ),
    ]
    for url, destination in candidates:
        if _download_file(
            url,
            destination,
            timeout=TIMEOUT_LARGE,
            min_bytes=1000,
        ):
            print(f"[D-A] FakeTrueBR disponível em {target}")
            return True
    print("[D-A] Não foi possível baixar FakeTrueBR.")
    if allow_demo:
        print(
            "[D-A] --allow-demo ativo: "
            "gerando dataset de demonstração."
        )
        _generate_demo_faketruebr(target)
        return True
    return False
def _generate_demo_faketruebr(target: Path) -> None:
    rows = [
        (
            "Notícia falsa de demonstração "
            "sobre política brasileira.",
            1,
        ),
        (
            "Notícia verdadeira de demonstração "
            "sobre política brasileira.",
            0,
        ),
    ]
    with open(
        target / "DEMO_faketruebr.csv",
        "w", newline="", encoding="utf-8",
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
# ── D-B — FakeNewsNet / PolitiFact ───────────────────────────────────────────
def download_fakenewsnet(
    out_dir: Path,
    *,
    allow_demo: bool = False,
) -> bool:
    """
    FakeNewsNet — subconjunto PolitiFact.
    Saída:
        data/raw/fakenewsnet/fake/
        data/raw/fakenewsnet/real/
    Nota sobre labels:
        A estrutura de diretórios (fake/ e real/) codifica o label.
        A conversão para inteiros 0/1 é responsabilidade do loader
        do pipeline, não deste downloader.
    """
    target   = out_dir / "fakenewsnet"
    fake_dir = target / "fake"
    real_dir = target / "real"
    if (
        _directory_has_files(fake_dir)
        and _directory_has_files(real_dir)
    ):
        print(
            f"[D-B] FakeNewsNet já presente em {target}. "
            f"fake={_count_files(fake_dir)} "
            f"real={_count_files(real_dir)}"
        )
        return True
    _ensure_dir(target)
    print("[D-B] Baixando FakeNewsNet / PolitiFact...")
    csv_urls: Dict[str, List[str]] = {
        "fake": [
            (
                "https://raw.githubusercontent.com/"
                "KaiDMML/FakeNewsNet/master/dataset/"
                "politifact_fake.csv"
            ),
            (
                "https://raw.githubusercontent.com/"
                "KaiDMML/FakeNewsNet/old_version/data/"
                "politifact_fake.csv"
            ),
        ],
        "real": [
            (
                "https://raw.githubusercontent.com/"
                "KaiDMML/FakeNewsNet/master/dataset/"
                "politifact_real.csv"
            ),
            (
                "https://raw.githubusercontent.com/"
                "KaiDMML/FakeNewsNet/old_version/data/"
                "politifact_real.csv"
            ),
        ],
    }
    downloaded: Dict[str, Path] = {}
    for label, urls in csv_urls.items():
        destination = target / f"politifact_{label}.csv"
        for url in urls:
            if _download_file(
                url,
                destination,
                timeout=TIMEOUT_DEFAULT,
                min_bytes=100,
            ):
                downloaded[label] = destination
                break
    if len(downloaded) == 2:
        if _convert_fakenewsnet_csvs(target, downloaded):
            print("[D-B] FakeNewsNet / PolitiFact concluído.")
            return True
    print("[D-B] Falha ao obter FakeNewsNet.")
    if allow_demo:
        print(
            "[D-B] --allow-demo ativo: "
            "gerando dataset de demonstração."
        )
        _generate_demo_fakenewsnet(target)
        return True
    return False

def _convert_fakenewsnet_csvs(
    target: Path,
    downloaded: Dict[str, Path],
) -> bool:
    """
    Converte os CSVs do FakeNewsNet em estrutura de diretórios:
        fake/<id>/news content.json
        real/<id>/news content.json
    Quando o campo 'text' estiver ausente ou vazio, usa o 'title'
    como fallback e registra um aviso com a contagem de casos.
    Isso permite que o loader identifique exemplos degenerados.
    """
    import csv as csv_mod
    for label, csv_path in downloaded.items():
        label_dir           = target / label
        _ensure_dir(label_dir)
        text_fallback_count = 0
        try:
            with open(
                csv_path,
                encoding="utf-8",
                errors="replace",
                newline="",
            ) as fh:
                reader = csv_mod.DictReader(fh)
                count  = 0
                for i, row in enumerate(reader):
                    news_id = (
                        row.get("id")
                        or row.get("news_id")
                        or f"{label}_{i}"
                    )
                    news_id  = (
                        str(news_id).strip() or f"{label}_{i}"
                    )
                    news_dir = label_dir / news_id
                    _ensure_dir(news_dir)
                    title = str(
                        row.get("title")
                        or row.get("headline")
                        or ""
                    ).strip()
                    text_raw = str(
                        row.get("text")
                        or row.get("article")
                        or row.get("body")
                        or ""
                    ).strip()
                    # Fallback: usa título quando texto está ausente
                    if not text_raw and title:
                        text_raw = title
                        text_fallback_count += 1
                    publish_date = str(
                        row.get("publish_date")
                        or row.get("date")
                        or ""
                    )
                    content = {
                        "title":        title,
                        "text":         text_raw,
                        "publish_date": publish_date,
                        "text_is_title_fallback": (
                            text_raw == title and bool(title)
                        ),
                    }
                    (news_dir / "news content.json").write_text(
                        json.dumps(
                            content,
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    count += 1
            print(
                f"      [OK] {label}: "
                f"{count} registros convertidos"
            )
            if text_fallback_count > 0:
                print(
                    f"      [AVISO] {label}: "
                    f"{text_fallback_count} registro(s) sem 'text'; "
                    f"campo preenchido com 'title' como fallback. "
                    f"O loader deve filtrar esses casos se necessário "
                    f"(campo 'text_is_title_fallback': true)."
                )
        except Exception as exc:
            print(
                f"      [ERRO] Conversão {csv_path.name}: {exc}"
            )
            return False
    return (
        _directory_has_files(target / "fake")
        and _directory_has_files(target / "real")
    )
def _generate_demo_fakenewsnet(target: Path) -> None:
    for label in ("fake", "real"):
        d = target / label / f"demo_{label}"
        _ensure_dir(d)
        content = {
            "title": f"Demo {label}",
            "text": (
                f"Notícia de demonstração "
                f"classificada como {label}."
            ),
            "publish_date":           "2020-01-01",
            "text_is_title_fallback": False,
        }
        (d / "news content.json").write_text(
            json.dumps(
                content, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
# ── D-C — PHEME ──────────────────────────────────────────────────────────────
def download_pheme(
    out_dir: Path,
    *,
    allow_demo: bool = False,
) -> bool:
    """
    PHEME.
    Tenta Figshare e, posteriormente, Hugging Face.
    Saída:
        data/raw/pheme/
    """
    target = out_dir / "pheme"
    if _directory_has_files(target):
        print(
            f"[D-C] PHEME já presente em {target} "
            f"({_count_files(target)} arquivos)."
        )
        return True
    _ensure_dir(target)
    print("[D-C] Baixando PHEME...")
    figshare_urls = [
        "https://figshare.com/ndownloader/files/11928496",
        "https://ndownloader.figshare.com/files/11928496",
    ]
    for url in figshare_urls:
        try:
            print(f"      Tentando Figshare: {url}")
            response = _request(
                url, timeout=TIMEOUT_LARGE, stream=True
            )
            content = b"".join(
                response.iter_content(chunk_size=1024 * 1024)
            )
            if len(content) < 10_000:
                print("      Conteúdo muito pequeno; ignorando.")
                continue
            if not content.startswith(b"PK"):
                print(
                    "      Conteúdo não é ZIP válido; ignorando."
                )
                continue
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                zf.extractall(target)
            if _directory_has_files(target):
                print("[D-C] PHEME baixado com sucesso.")
                return True
        except Exception as exc:
            print(f"      Falha Figshare: {exc}")
    print("[D-C] Figshare falhou; tentando Hugging Face...")
    try:
        from datasets import load_dataset  # type: ignore[import]
        for hf_id in ("mehdialam/pheme", "tner/pheme"):
            try:
                print(f"      Tentando Hugging Face: {hf_id}")
                ds = load_dataset(hf_id)
                _pheme_from_hf(ds, target)
                if _directory_has_files(target):
                    return True
            except Exception as exc:
                print(
                    f"      {hf_id} falhou: "
                    f"{str(exc)[:200]}"
                )
    except ImportError:
        print(
            "[D-C] Biblioteca 'datasets' não instalada; "
            "pulando fallback Hugging Face."
        )
    if allow_demo:
        print("[D-C] --allow-demo ativo.")
        _generate_demo_pheme(target)
        return True
    print("[D-C] Não foi possível baixar PHEME.")
    return False
def _pheme_from_hf(ds, target: Path) -> None:
    import itertools
    count = 0
    for split in ds.keys():
        for row in itertools.islice(ds[split], 10_000):
            event    = str(row.get("event", "unknown_event"))
            veracity = str(
                row.get("veracity", "unverified")
            ).lower()
            rtype = (
                "rumours"
                if veracity in ("true", "false")
                else "non-rumours"
            )
            thread_id = str(
                row.get("thread_id", row.get("id", count))
            )
            tweet_dir = (
                target
                / event
                / rtype
                / thread_id
                / "source-tweets"
            )
            _ensure_dir(tweet_dir)
            tweet_data = {
                "text": str(row.get("text", "")),
                "created_at": str(
                    row.get(
                        "created_at",
                        "Mon Jan 01 12:00:00 +0000 2015",
                    )
                ),
            }
            (tweet_dir / f"{thread_id}.json").write_text(
                json.dumps(
                    tweet_data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if rtype == "rumours":
                (
                    target
                    / event
                    / rtype
                    / thread_id
                    / "annotation.json"
                ).write_text(
                    json.dumps(
                        {"veracity": veracity},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            count += 1
    print(
        f"      [OK] PHEME construído via HF: "
        f"{count} registros"
    )
def _generate_demo_pheme(target: Path) -> None:
    d = (
        target
        / "demo_event"
        / "rumours"
        / "demo_0001"
        / "source-tweets"
    )
    _ensure_dir(d)
    (d / "demo_0001.json").write_text(
        json.dumps(
            {
                "text": "PHEME demo",
                "created_at": (
                    "Mon Jan 01 12:00:00 +0000 2015"
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (
        target
        / "demo_event"
        / "rumours"
        / "demo_0001"
        / "annotation.json"
    ).write_text(
        json.dumps({"veracity": "false"}),
        encoding="utf-8",
    )
# ── D-D — Fake.br-Corpus ─────────────────────────────────────────────────────
def download_fakebr(
    out_dir: Path,
    *,
    allow_demo: bool = False,
) -> bool:
    """
    Fake.br-Corpus.
    Prioridade:
        1. GitHub original (CSVs separados por classe)
        2. ZIP do repositório completo
    Saída:
        data/raw/fakebr/fakebr.csv
    Labels numéricos:
        1 = fake (notícia falsa)
        0 = true (notícia verdadeira)
    ATENÇÃO: a conversão de labels textuais para numéricos é aplicada
    aqui apenas se confirmado que o loader do pipeline espera inteiros.
    Caso o loader faça a conversão internamente, comentar a chamada
    a _normalize_labels() para evitar dupla transformação.
    Verificar constante LABEL_MAP_FAKEBR no topo deste arquivo.
    """
    target = out_dir / "fakebr"
    _ensure_dir(target)
    existing = (
        list(target.glob("*.csv"))
        + list(target.glob("*.parquet"))
    )
    if existing or _directory_has_files(target / "full_texts"):
        print(f"[D-D] Fake.br já presente em {target}.")
        return True
    print("[D-D] Baixando Fake.br-Corpus...")
    github_files = {
        "fake_texts.csv": (
            "https://raw.githubusercontent.com/"
            "roneysco/Fake.br-Corpus/master/"
            "size_normalized_texts/fake_texts.csv"
        ),
        "true_texts.csv": (
            "https://raw.githubusercontent.com/"
            "roneysco/Fake.br-Corpus/master/"
            "size_normalized_texts/true_texts.csv"
        ),
    }
    frames = []
    pd_mod = _try_import_pandas()
    if pd_mod is not None:
        for filename, url in github_files.items():
            try:
                response = _request(url, timeout=TIMEOUT_LARGE)
                if len(response.content) < 500:
                    continue
                df = pd_mod.read_csv(
                    io.StringIO(
                        response.content.decode(
                            "utf-8", errors="replace"
                        )
                    )
                )
                if df.empty:
                    continue
                label_str = (
                    "fake"
                    if filename.startswith("fake_")
                    else "true"
                )
                df["label"] = label_str
                frames.append(df)
                print(
                    f"      [OK] {filename}: "
                    f"{len(df)} linhas"
                )
            except Exception as exc:
                print(f"      Falha {filename}: {exc}")
    if frames:
        combined = pd_mod.concat(frames, ignore_index=True)
        # Converte labels textuais para numéricos.
        # ATENÇÃO: verificar se o loader do pipeline já faz essa
        # conversão antes de habilitar esta linha.
        combined = _normalize_labels(
            combined,
            label_col="label",
            label_map=LABEL_MAP_FAKEBR,
            dataset_name="Fake.br",
        )
        destination = target / "fakebr.csv"
        combined.to_csv(
            destination, index=False, encoding="utf-8"
        )
        print(
            f"[D-D] Fake.br concluído: "
            f"{len(combined)} linhas."
        )
        print(
            f"      Labels: "
            f"{combined['label'].value_counts().to_dict()}"
        )
        return True
    # Fallback: ZIP do repositório
    zip_urls = [
        (
            "https://github.com/roneysco/Fake.br-Corpus/"
            "archive/refs/heads/master.zip"
        ),
        (
            "https://github.com/roneysco/Fake.br-Corpus/"
            "archive/refs/heads/main.zip"
        ),
    ]
    for url in zip_urls:
        try:
            response = _request(url, timeout=TIMEOUT_LARGE)
            if not response.content.startswith(b"PK"):
                continue
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / "fakebr.zip"
                archive.write_bytes(response.content)
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(tmp)
                roots = [
                    p for p in Path(tmp).iterdir()
                    if p.is_dir()
                ]

                source_root = next(
                    (
                        p for p in roots
                        if (
                            p / "size_normalized_texts"
                        ).exists()
                    ),
                    None,
                )
                if source_root is not None:
                    shutil.copytree(
                        source_root / "size_normalized_texts",
                        target / "size_normalized_texts",
                        dirs_exist_ok=True,
                    )
                    if _directory_has_files(
                        target / "size_normalized_texts"
                    ):
                        print(
                            "[D-D] Fake.br baixado via GitHub ZIP.\n"
                            "      AVISO: labels não convertidos para "
                            "inteiros neste modo de download. "
                            "O loader do pipeline deve realizar a "
                            "conversão 'fake'->1, 'true'->0."
                        )
                        return True
        except Exception as exc:
            print(f"      GitHub ZIP falhou: {exc}")
    if allow_demo:
        print("[D-D] --allow-demo ativo.")
        _generate_demo_fakebr(target)
        return True
    print("[D-D] Não foi possível baixar Fake.br.")
    return False
def _generate_demo_fakebr(target: Path) -> None:
    with open(
        target / "DEMO_fakebr.csv",
        "w", newline="", encoding="utf-8",
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        writer.writerow(["Notícia fake de demonstração.", 1])
        writer.writerow(["Notícia real de demonstração.", 0])
# ── D-E — FakeRecogna ────────────────────────────────────────────────────────
def download_fakerecogna(
    out_dir: Path,
    *,
    allow_demo: bool = False,
) -> bool:
    """
    FakeRecogna original.
    Fonte primária : GitHub — Gabriel-Lino-Garcia/FakeRecogna
    Fallback       : Hugging Face — recogna-nlp/FakeRecogna
    Saída:
        data/raw/fakerecogna/FakeRecogna.xlsx  (preferencial)
        data/raw/fakerecogna/FakeRecogna.csv   (fallback HF)
    """
    target = out_dir / "fakerecogna"
    _ensure_dir(target)
    if (
        (target / "FakeRecogna.xlsx").exists()
        or (target / "FakeRecogna.csv").exists()
        or _directory_has_files(target / "raw")
    ):
        print(f"[D-E] FakeRecogna já presente em {target}.")
        return True
    print("[D-E] Baixando FakeRecogna...")
    github_urls = [
        (
            "https://raw.githubusercontent.com/"
            "Gabriel-Lino-Garcia/FakeRecogna/master/"
            "dataset/FakeRecogna.xlsx",
            target / "FakeRecogna.xlsx",
        ),
        (
            "https://raw.githubusercontent.com/"
            "Gabriel-Lino-Garcia/FakeRecogna/main/"
            "dataset/FakeRecogna.xlsx",
            target / "FakeRecogna.xlsx",
        ),
    ]
    for url, destination in github_urls:
        if _download_file(
            url,
            destination,
            timeout=TIMEOUT_LARGE,
            min_bytes=100_000,
        ):
            print("[D-E] FakeRecogna original baixado.")
            return True
    hf_urls = [
        (
            "https://huggingface.co/datasets/"
            "recogna-nlp/FakeRecogna/resolve/main/"
            "FakeRecogna.csv?download=true",
            target / "FakeRecogna.csv",
        ),
    ]
    for url, destination in hf_urls:
        if _download_file(
            url,
            destination,
            timeout=TIMEOUT_LARGE,
            min_bytes=100_000,
        ):
            print("[D-E] FakeRecogna obtido via Hugging Face.")
            return True
    if allow_demo:
        print("[D-E] --allow-demo ativo.")
        _generate_demo_fakerecogna(target)
        return True
    print("[D-E] Não foi possível baixar FakeRecogna.")
    return False
def _generate_demo_fakerecogna(target: Path) -> None:
    """
    Labels: 1 = fake, 0 = real — padrão do projeto.
    """
    with open(
        target / "DEMO_fakerecogna.csv",
        "w", newline="", encoding="utf-8",
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Title", "Sub-title", "News",
            "Category", "Author", "Date", "URL", "Class",
        ])
        writer.writerow([
            "Notícia fake demo", "",
            "Texto falso de demonstração.",
            "Politics", "", "2020-01-01", "", 1,  # 1 = fake
        ])
        writer.writerow([
            "Notícia real demo", "",
            "Texto verdadeiro de demonstração.",
            "Politics", "", "2020-01-02", "", 0,  # 0 = real
        ])
# ── D-F — CONSTRAINT COVID-19 ────────────────────────────────────────────────
def download_constraint_covid19(
    out_dir: Path,
    *,
    allow_demo: bool = False,
) -> bool:
    """
    CONSTRAINT 2021 COVID-19 Fake News.
    Tenta:
        1. Hugging Face
        2. GitHub raw
    Saída:
        data/raw/constraint_covid19/train.csv
        data/raw/constraint_covid19/val.csv
        data/raw/constraint_covid19/test.csv
    """
    target = out_dir / "constraint_covid19"
    _ensure_dir(target)
    if (
        (target / "train.csv").exists()
        and (
            (target / "test.csv").exists()
            or (target / "val.csv").exists()
        )
    ):
        print(f"[D-F] CONSTRAINT já presente em {target}.")
        return True
    print("[D-F] Baixando CONSTRAINT COVID-19...")
    try:
        from datasets import load_dataset  # type: ignore[import]
        hf_ids = [
            "covid_fake_news",
            "RUCAIBox/covid-fake-news",
        ]
        for hf_id in hf_ids:
            try:
                print(f"      Hugging Face: {hf_id}")
                ds      = load_dataset(hf_id)
                written = 0
                for split_name, split_data in ds.items():
                    destination = target / f"{split_name}.csv"
                    split_data.to_csv(
                        str(destination), index=False
                    )
                    if destination.stat().st_size > 100:
                        written += 1
                    print(
                        f"      [OK] {destination.name} "
                        f"({len(split_data)} linhas)"
                    )
                if written >= 2:
                    print(
                        "[D-F] CONSTRAINT COVID-19 "
                        "concluído via Hugging Face."
                    )
                    return True
            except Exception as exc:
                print(
                    f"      {hf_id} falhou: "
                    f"{str(exc)[:200]}"
                )
    except ImportError:
        print("      Biblioteca 'datasets' não instalada.")
    # Fallback GitHub
    github_urls: Dict[str, str] = {
        "train": (
            "https://raw.githubusercontent.com/"
            "diptamath/covid_fake_news/main/data/"
            "Constraint_Train.csv"
        ),
        "val": (
            "https://raw.githubusercontent.com/"
            "diptamath/covid_fake_news/main/data/"
            "Constraint_Val.csv"
        ),
        "test": (
            "https://raw.githubusercontent.com/"
            "diptamath/covid_fake_news/main/data/"
            "Constraint_Test.csv"
        ),
    }
    downloaded = 0
    for split, url in github_urls.items():
        destination = target / f"{split}.csv"
        if _download_file(
            url,
            destination,
            timeout=TIMEOUT_LARGE,
            min_bytes=100,
        ):
            downloaded += 1
    if downloaded >= 2:
        print("[D-F] CONSTRAINT COVID-19 concluído via GitHub.")
        return True
    if allow_demo:
        print("[D-F] --allow-demo ativo.")
        _generate_demo_constraint(target)
        return True
    print("[D-F] Não foi possível baixar CONSTRAINT COVID-19.")
    return False
def _generate_demo_constraint(target: Path) -> None:
    rows = [
        ("Notícia falsa de COVID de demonstração.", "fake"),
        ("Notícia verdadeira de COVID de demonstração.", "real"),
    ]
    with open(
        target / "DEMO_constraint.csv",
        "w", newline="", encoding="utf-8",
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["tweet", "label"])
        writer.writerows(rows)
# ── Dispatcher ────────────────────────────────────────────────────────────────
DOWNLOADERS: Dict[str, Callable[..., bool]] = {
    "da": download_faketruebr,
    "db": download_fakenewsnet,
    "dc": download_pheme,
    "dd": download_fakebr,
    "de": download_fakerecogna,
    "df": download_constraint_covid19,
}
# ── Validação final ───────────────────────────────────────────────────────────
def validate_download(flag: str, data_dir: Path) -> bool:
    """
    Verifica se o dataset deixou artefatos úteis no diretório esperado.
    Não considera arquivos .part como download concluído.
    Emite aviso quando o diretório contém apenas dados de demonstração
    (arquivos DEMO_*) gerados por --allow-demo.
    A validação semântica completa é responsabilidade dos loaders
    do pipeline.
    """
    spec   = DATASET_SPECS[flag]
    target = data_dir / spec["directory"]
    if not target.exists():
        return False
    if not _directory_has_files(target):
        return False
    # Arquivo .part indica download incompleto
    if any(target.rglob("*.part")):
        print(
            f"[AVISO] {flag}: arquivo(s) .part encontrado(s) em "
            f"{target}. Download pode estar incompleto."
        )
        return False
    # Detecta dados sintéticos gerados por --allow-demo
    demo_files = list(target.rglob("DEMO_*"))
    if demo_files:
        print(
            f"[AVISO] {flag}: diretório contém apenas dados de "
            f"demonstração ({len(demo_files)} arquivo(s) DEMO_*). "
            f"NÃO usar em experimentos da monografia."
        )
    return True
# ── CLI ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Baixar os 6 datasets da monografia: "
            "FakeTrueBR, Fake.br, FakeRecogna, "
            "FakeNewsNet/PolitiFact, PHEME e "
            "CONSTRAINT COVID-19."
        )
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(DATASET_SPECS.keys()),
        default=list(DATASET_SPECS.keys()),
        metavar="{da,db,dc,dd,de,df}",
        help="Datasets a baixar. Padrão: todos os 6.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=(
            f"Diretório de destino. "
            f"Padrão: {DEFAULT_DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--allow-demo",
        action="store_true",
        help=(
            "Gera dataset sintético de demonstração "
            "quando o download falhar. "
            "NÃO usar em experimentos da monografia."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help=(
            "Lista o mapeamento dos 6 datasets e encerra."
        ),
    )
    return parser
def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    if args.list:
        _print_dataset_map()
        return 0
    data_dir = Path(args.data_dir).expanduser().resolve()
    _ensure_dir(data_dir)
    _print_header(
        "Download dos 6 datasets — estudo de Fake News"
    )
    print(f"Raiz do projeto : {ROOT}")
    print(f"Destino         : {data_dir}")
    print(f"Datasets        : {', '.join(args.only)}")
    print(f"Demo permitido  : {args.allow_demo}")
    print()
    _print_dataset_map()
    results: Dict[str, bool] = {}
    for flag in args.only:
        spec       = DATASET_SPECS[flag]
        downloader = DOWNLOADERS[flag]
        print()
        print("-" * 72)
        print(
            f"[{flag.upper()}] "
            f"{spec['name']} "
            f"-> {spec['directory']}/"
        )
        print("-" * 72)
        try:
            ok = downloader(
                data_dir,
                allow_demo=args.allow_demo,
            )
        except KeyboardInterrupt:
            print("\n[ABORTADO] Interrompido pelo usuário.")
            return 130
        except Exception as exc:
            print(
                f"[ERRO] Exceção não tratada em {flag}: "
                f"{exc}"
            )
            ok = False
        if ok:
            ok = validate_download(flag, data_dir)
        results[flag] = ok
        print(
            f"[{flag.upper()}] Resultado: "
            f"{'OK' if ok else 'FALHA'}"
        )
    # Resumo
    _print_header("Resumo")
    failed: List[str] = []
    for flag in args.only:
        spec   = DATASET_SPECS[flag]
        ok     = results[flag]
        status = "[OK]  " if ok else "[FALHA]"
        print(
            f"{status} "
            f"{flag.upper():<2} "
            f"{spec['canonical']:<20} "
            f"{data_dir / spec['directory']}"
        )
        if not ok:
            failed.append(flag)
    print()
    if failed:
        print(
            "[FALHA] Um ou mais datasets não foram obtidos:"
        )
        for flag in failed:
            print(
                f"        {flag} -> "
                f"{DATASET_SPECS[flag]['name']}"
            )
        print()
        print(
            "O script NÃO considera a execução bem-sucedida."
        )
        print(
            "Verifique as mensagens acima e execute novamente."
        )
        return 1
    print(
        "[OK] Todos os datasets solicitados foram "
        "baixados e passaram pela validação básica."
    )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())