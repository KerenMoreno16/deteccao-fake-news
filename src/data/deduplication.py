"""
deduplication.py — Exact and near-duplicate detection across
and within datasets.
This module is a methodological differentiator of this research:
it detects data leakage from duplicates before any train/test
split is performed.
Correções aplicadas:
  - find_near_duplicates(): aviso de memória emitido ANTES
    da alocação da matriz de similaridade.
  - sample_size fixo com random_state=42 documentado como
    intencional (deduplicação não é experimento de aprendizado).
  - _compute_tfidf_hashes() mantida como função privada.
  - Loop O(n²) substituído por operação numpy vetorizada.
"""
import hashlib
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from src.utils.logging import get_logger
from scipy.sparse import spmatrix, coo_matrix
from scipy.sparse.csgraph import connected_components
logger = get_logger(__name__)
# ---------------------------------------------------------------------------
# Exact duplicate detection
# ---------------------------------------------------------------------------
def text_hash(text: str) -> str:
    """
    SHA-256 digest of normalized text (lowercase, stripped).
    A normalização garante que textos idênticos com diferenças
    de capitalização ou espaços sejam tratados como duplicatas.
    """
    normalized = text.lower().strip()
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()
def mark_exact_duplicates(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Adiciona coluna booleana 'is_duplicate' — True para todas
    as ocorrências exceto a primeira de cada texto único.
    Parâmetros
    ----------
    df       : DataFrame com coluna de texto.
    text_col : nome da coluna de texto.
    Retorna
    -------
    DataFrame com coluna 'is_duplicate' adicionada.
    O DataFrame original não é modificado (operação em cópia).
    """
    if text_col not in df.columns:
        raise ValueError(
            f"Coluna '{text_col}' não encontrada. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )
    df = df.copy()
    df["_text_hash"]   = df[text_col].fillna("").apply(text_hash)
    df["is_duplicate"] = df.duplicated(
        subset=["_text_hash"], keep="first"
    )
    n_dups = int(df["is_duplicate"].sum())
    logger.info(
        "Exact duplicates: %d / %d (%.1f%%)",
        n_dups,
        len(df),
        100.0 * n_dups / len(df) if len(df) else 0.0,
    )
    df = df.drop(columns=["_text_hash"])
    return df
def remove_exact_duplicates(
    df: pd.DataFrame,
    text_col: str = "text",
) -> pd.DataFrame:
    """
    Remove textos duplicados mantendo a primeira ocorrência.
    Parâmetros
    ----------
    df       : DataFrame com coluna de texto.
    text_col : nome da coluna de texto.
    Retorna
    -------
    DataFrame sem duplicatas exatas, com índice resetado.
    """
    df     = mark_exact_duplicates(df, text_col)
    before = len(df)
    df = (
        df[~df["is_duplicate"]]
        .drop(columns=["is_duplicate"])
        .reset_index(drop=True)
    )
    removed = before - len(df)
    if removed > 0:
        logger.info(
            "Removed %d exact duplicates (%d -> %d rows)",
            removed, before, len(df),
        )
    else:
        logger.info(
            "No exact duplicates found (%d rows unchanged)",
            len(df),
        )
    return df
# ---------------------------------------------------------------------------
# Cross-dataset leakage detection
# ---------------------------------------------------------------------------
def detect_cross_dataset_leakage(
    datasets: Dict[str, pd.DataFrame],
    text_col: str = "text",
) -> Dict[Tuple[str, str], int]:
    """
    Para cada par de datasets, conta e registra sobreposições
    exatas de texto.
    Esta verificação deve ser executada ANTES de qualquer
    divisão treino/teste para garantir que não haja vazamento
    de dados entre corpora.
    Parâmetros
    ----------
    datasets : dicionário {nome: DataFrame}.
    text_col : nome da coluna de texto.
    Retorna
    -------
    dict {(nome_a, nome_b): contagem_sobreposição}.
    """
    # Verifica coluna de texto em todos os datasets
    for name, df in datasets.items():
        if text_col not in df.columns:
            raise ValueError(
                f"Dataset '{name}': coluna '{text_col}' "
                f"não encontrada."
            )
    hashes: Dict[str, set] = {}
    for name, df in datasets.items():
        hashes[name] = set(
            df[text_col].fillna("").apply(text_hash)
        )
        logger.info(
            "Cross-dataset leakage: '%s' -> %d hashes únicos",
            name, len(hashes[name]),
        )
    leakage: Dict[Tuple[str, str], int] = {}
    names = list(datasets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b    = names[i], names[j]
            overlap = len(hashes[a] & hashes[b])
            leakage[(a, b)] = overlap
            if overlap > 0:
                logger.warning(
                    "Cross-dataset overlap %s ↔ %s: "
                    "%d texto(s) em comum. "
                    "Remover antes do split treino/teste.",
                    a, b, overlap,
                )
            else:
                logger.info(
                    "Cross-dataset overlap %s ↔ %s: "
                    "nenhuma sobreposição.",
                    a, b,
                )
    return leakage
# ---------------------------------------------------------------------------
# Near-duplicate detection (cosine similarity on hashed TF-IDF)
# ---------------------------------------------------------------------------
def _compute_tfidf_hashes(
    texts: pd.Series,
    n_features: int = 2 ** 18,
) -> "spmatrix":
    """
    Projeta textos em espaço de features fixo usando
    HashingVectorizer com bigramas.
    Função privada — usada apenas por find_near_duplicates().
    Parâmetros
    ----------
    texts      : Series de textos.
    n_features : dimensão do espaço hash (padrão: 2^18).
    Retorna
    -------
    Matriz esparsa (n_texts, n_features) normalizada L2.
    """
    vec = HashingVectorizer(
        n_features=n_features,
        ngram_range=(1, 2),
        norm="l2",
        alternate_sign=False,
    )
    return vec.transform(texts.fillna(""))
def find_near_duplicates(
    df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = 0.95,
    sample_size: Optional[int] = None,
    block_size: int = 2000,
) -> pd.DataFrame:
    """
    Identifica pares de textos quase-duplicados usando
    similaridade de cosseno sobre features TF-IDF hash.
    Devido à complexidade quadrática O(n²), esta função
    amostra *sample_size* linhas quando o dataset é maior,
    emitindo aviso.
    O aviso de uso de memória é emitido ANTES da alocação
    da matriz para permitir cancelamento pelo usuário.
    Parâmetros
    ----------
    df          : DataFrame com coluna de texto.
    text_col    : nome da coluna de texto.
    threshold   : limiar de similaridade de cosseno
                  (padrão: 0.95).
    sample_size : máximo de textos analisados.
                  None = sem limite (cuidado com OOM).
                  Padrão: None (processa todos os textos).
    Retorna
    -------
    DataFrame com colunas ['idx_a', 'idx_b', 'similarity'],
    ordenado por similaridade decrescente.
    Retorna DataFrame vazio se nenhum par encontrado.
    Notas
    -----
    - random_state=42 fixo na amostragem é intencional:
      deduplicação não é experimento de aprendizado e
      a consistência entre chamadas é mais importante
      que a variação entre seeds experimentais.
    - Para detecção exaustiva, usar remove_exact_duplicates()
      em conjunto com esta função.
    """
    if text_col not in df.columns:
        raise ValueError(
            f"Coluna '{text_col}' não encontrada. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )
    if not (0.0 < threshold <= 1.0):
        raise ValueError(
            f"threshold deve estar em (0, 1]. "
            f"Recebido: {threshold}"
        )
    texts = df[text_col].reset_index(drop=True)
    n_original = len(texts)
    # Amostragem se necessário
    if sample_size is not None and len(texts) > sample_size:
        logger.warning(
            "Near-duplicate detection: amostrando %d / %d "
            "textos (varredura completa é O(n²)). "
            "random_state=42 fixo para consistência.",
            sample_size,
            len(texts),
        )
        texts = texts.sample(
            n=sample_size,
            random_state=42,
        ).reset_index(drop=True)
    n = len(texts)
    logger.info(
        "Near-duplicate detection: %d textos | varredura em blocos de %d "
        "(memória O(block*n), sem amostragem).", n, block_size,
    )
    # Vetorização (linhas já normalizadas L2 → produto escalar = cosseno)
    X = _compute_tfidf_hashes(texts).tocsr()
    XT = X.T.tocsc()
    a_parts, b_parts, s_parts = [], [], []
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = (X[start:stop] @ XT).tocoo()
        keep = (block.data >= threshold) & ((block.row + start) < block.col)
        if keep.any():
            a_parts.append(block.row[keep] + start)
            b_parts.append(block.col[keep])
            s_parts.append(block.data[keep])
    if a_parts:
        idx_a = np.concatenate(a_parts)
        idx_b = np.concatenate(b_parts)
        similarities = np.concatenate(s_parts)
    else:
        idx_a = idx_b = np.array([], dtype=int)
        similarities = np.array([], dtype=float)
    if len(idx_a) == 0:
        logger.info(
            "Near-duplicates (threshold=%.2f): "
            "nenhum par encontrado em amostra de %d / %d textos.",
            threshold, n, n_original,
        )
        return pd.DataFrame(
            columns=["idx_a", "idx_b", "similarity"]
        )
    result = pd.DataFrame(
        {
            "idx_a":      idx_a.astype(int),
            "idx_b":      idx_b.astype(int),
            "similarity": similarities.astype(float),
        }
    ).sort_values(
        "similarity",
        ascending=False,
    ).reset_index(drop=True)
    logger.info(
        "Near-duplicates (threshold=%.2f): "
        "%d par(es) encontrado(s) em amostra de %d / %d textos.",
        threshold, len(result), n, n_original,
    )
    return result


def near_duplicate_groups(
    df: pd.DataFrame,
    text_col: str = "text",
    similarity_threshold: float = 0.90,
) -> np.ndarray:
    """
    Rótulo de grupo (componente conexo) para cada linha de *df*; linhas sem
    quase-duplicata formam grupo próprio. Útil para GroupShuffleSplit.
    """

    n = len(df)
    pairs = find_near_duplicates(
        df.reset_index(drop=True), text_col=text_col,
        threshold=similarity_threshold, sample_size=None,
    )
    if pairs.empty:
        return np.arange(n)
    i = pairs["idx_a"].to_numpy()
    j = pairs["idx_b"].to_numpy()
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels


def remove_near_duplicates(
    df: pd.DataFrame,
    text_col: str = "text",
    similarity_threshold: float = 0.90,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Remove quase-duplicatas (cosseno em TF-IDF hash >= limiar), mantendo a primeira
    ocorrência de cada componente conexo. Deve rodar ANTES do split, para que
    nenhum par quase-idêntico atravesse treino/teste (Lee et al., 2022).
    """
    df = df.reset_index(drop=True)
    grp = near_duplicate_groups(df, text_col, similarity_threshold)
    keep = ~pd.Series(grp).duplicated(keep="first").to_numpy()
    if label_col in df.columns:
        n_conf = int(
            (pd.DataFrame({"g": grp, "y": df[label_col].to_numpy()})
             .groupby("g")["y"].nunique() > 1).sum()
        )
        if n_conf:
            logger.warning(
                "Near-dup: %d grupo(s) com rótulos conflitantes (mantida a 1ª ocorrência).",
                n_conf,
            )
    logger.info(
        "Near-duplicates removidos (limiar=%.2f): "
        "%d -> %d (removidos %d)",
        similarity_threshold,
        len(df),
        int(keep.sum()),
        int((~keep).sum()),
    )
    return df[keep].reset_index(drop=True)