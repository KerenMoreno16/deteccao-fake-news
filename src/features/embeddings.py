"""
embeddings.py — Word embedding utilities for BiLSTM.
Suporta:
  .bin — FastText binário com subwords (requer fasttext)
  .vec — FastText texto sem dependências extras
Correções aplicadas:
  - Duplicação do arquivo removida (duas definições
    de Vocabulary e imports).
  - _load_bin() recebe seed como parâmetro para
    reprodutibilidade na redimensionalização.
  - collate_batch() com validação de formato do batch.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from src.utils.logging import get_logger
logger = get_logger(__name__)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX   = 0
UNK_IDX   = 1
# Import defensivo do pacote fasttext
try:
    import fasttext as _ft_lib
    _HAS_FASTTEXT = True
except ImportError:
    _HAS_FASTTEXT = False
    logger.warning(
        "Pacote 'fasttext' não instalado. "
        "Arquivos .bin não serão carregados. "
        "Instale: pip install fasttext-wheel"
    )
# ---------------------------------------------------------------------------
# Vocabulário
# ---------------------------------------------------------------------------
class Vocabulary:
    """Token-to-index mapping built from training data."""
    def __init__(self, min_freq: int = 2):
        self.min_freq   = min_freq
        self.token2idx: Dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
        }
        self.idx2token: Dict[int, str] = {
            PAD_IDX: PAD_TOKEN,
            UNK_IDX: UNK_TOKEN,
        }
        self._freq: Dict[str, int] = {}
    def build(self, texts: List[str]) -> None:
        """
        Constrói vocabulário a partir de textos de treino.
        Parâmetros
        ----------
        texts : lista de textos já limpos
                (whitespace-tokenizados).
        Notas
        -----
        Deve ser chamado APENAS com textos de treino
        para evitar vazamento de vocabulário.
        Tokens com freq < min_freq são excluídos.
        """
        for text in texts:
            for token in text.split():
                self._freq[token] = (
                    self._freq.get(token, 0) + 1
                )
        for token, freq in self._freq.items():
            if (
                freq >= self.min_freq
                and token not in self.token2idx
            ):
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx]   = token
        logger.info(
            "Vocabulary built: %d tokens "
            "(min_freq=%d)",
            len(self.token2idx), self.min_freq,
        )
    def encode(
        self,
        text: str,
        max_len: Optional[int] = None,
    ) -> List[int]:
        """
        Converte texto em lista de índices.
        Tokens fora do vocabulário recebem UNK_IDX.
        """
        tokens = text.split()
        if max_len is not None:
            tokens = tokens[:max_len]
        return [
            self.token2idx.get(t, UNK_IDX)
            for t in tokens
        ]
    def __len__(self) -> int:
        return len(self.token2idx)
# ---------------------------------------------------------------------------
# Carregamento de embeddings FastText
# ---------------------------------------------------------------------------
def load_fasttext_embeddings(
    fasttext_path: str,
    vocab: Vocabulary,
    embedding_dim: int = 300,
    seed: int = 42,
) -> Optional[torch.Tensor]:
    """
    Carrega embeddings FastText para o vocabulário.
    Suporta dois formatos:
      .bin — binário com subwords (cobertura ~100%)
      .vec — texto sem dependências (cobertura ~40-60%)
    Parâmetros
    ----------
    fasttext_path : caminho para .bin ou .vec.
    vocab         : Vocabulary com token2idx.
    embedding_dim : dimensão esperada dos embeddings.
    seed          : seed para inicialização aleatória
                    de tokens OOV.
    Retorna
    -------
    torch.Tensor de shape (vocab_size, embedding_dim)
    ou None se o arquivo não existir.
    """
    path   = Path(fasttext_path)
    suffix = path.suffix.lower()
    if not path.exists():
        logger.warning(
            "FastText não encontrado: %s. "
            "BiLSTM usará embeddings aleatórios.",
            fasttext_path,
        )
        return None
    # Inicializar matriz com valores aleatórios
    rng    = np.random.default_rng(seed)
    matrix = rng.normal(
        scale=0.01,
        size=(len(vocab), embedding_dim),
    ).astype(np.float32)
    # PAD = vetor zero (não contribui para o modelo)
    matrix[PAD_IDX] = 0.0
    if suffix == ".bin":
        return _load_bin(
            path, vocab, matrix,
            embedding_dim, seed=seed,
        )
    if suffix in (".vec", ".txt"):
        return _load_vec(
            path, vocab, matrix, embedding_dim,
        )
    logger.warning(
        "Formato não reconhecido: '%s'. "
        "Use .bin ou .vec.",
        suffix,
    )
    return None
def _load_bin(
    path: Path,
    vocab: Vocabulary,
    matrix: np.ndarray,
    embedding_dim: int,
    seed: int = 42,
) -> Optional[torch.Tensor]:
    """
    Carrega embeddings de arquivo .bin via subwords.
    Requer: pip install fasttext-wheel
    CORRIGIDO: seed recebido como parâmetro para
    reprodutibilidade na redimensionalização.
    """
    if not _HAS_FASTTEXT:
        logger.warning(
            "Arquivo .bin requer pacote fasttext. "
            "Instale: pip install fasttext-wheel  "
            "ou use arquivo .vec. "
            "Treinando com embeddings aleatórios."
        )
        return torch.tensor(matrix)
    try:
        logger.info(
            "Carregando FastText .bin: %s", path
        )
        ft_model   = _ft_lib.load_model(str(path))
        actual_dim = ft_model.get_dimension()
        if actual_dim != embedding_dim:
            logger.warning(
                "FastText dim=%d difere de "
                "embedding_dim=%d. Usando %d.",
                actual_dim, embedding_dim, actual_dim,
            )
            # CORRIGIDO: usar seed do parâmetro
            rng    = np.random.default_rng(seed)
            matrix = rng.normal(
                scale=0.01,
                size=(len(vocab), actual_dim),
            ).astype(np.float32)
            matrix[PAD_IDX] = 0.0
            embedding_dim   = actual_dim
        # UNK = média dos vetores mais comuns
        # (cobertura do vocabulário geral do modelo)
        sample_words = ft_model.get_words()[:1000]
        if sample_words:
            unk_vec = np.mean(
                [
                    ft_model.get_word_vector(w)
                    for w in sample_words
                ],
                axis=0,
            ).astype(np.float32)
            matrix[UNK_IDX] = unk_vec
        found      = 0
        not_found  = 0
        vocab_size = len(vocab)
        for token, idx in vocab.token2idx.items():
            if token in (PAD_TOKEN, UNK_TOKEN):
                continue
            if idx >= vocab_size:
                continue
            try:
                vec = np.array(
                    ft_model.get_word_vector(token),
                    dtype=np.float32,
                )
                # Vetor zero indica ausência de subwords
                if np.linalg.norm(vec) > 0:
                    matrix[idx] = vec
                    found += 1
                else:
                    not_found += 1
            except Exception as e:
                logger.debug(
                    "get_word_vector falhou para "
                    "'%s': %s",
                    token, e,
                )
                not_found += 1
        n_tokens = vocab_size - 2
        coverage = (
            100.0 * found / n_tokens
            if n_tokens > 0
            else 0.0
        )
        logger.info(
            "FastText .bin: %d/%d tokens (%.1f%%) | "
            "%d sem subwords",
            found, n_tokens,
            coverage, not_found,
        )
        if coverage < 10.0:
            logger.warning(
                "FastText cobertura muito baixa "
                "(%.1f%%). Verificar idioma do .bin.",
                coverage,
            )
        return torch.tensor(matrix)
    except Exception as e:
        logger.error(
            "_load_bin falhou: %s — "
            "usando embeddings aleatórios.", e,
        )
        return torch.tensor(matrix)
def _load_vec(
    path: Path,
    vocab: Vocabulary,
    matrix: np.ndarray,
    embedding_dim: int,
) -> Optional[torch.Tensor]:
    """
    Carrega embeddings de arquivo .vec (formato texto).
    Não requer pacote fasttext.
    Cobertura menor que .bin (~40-60%) pois não usa
    subwords.
    Parâmetros
    ----------
    path          : caminho para o arquivo .vec.
    vocab         : Vocabulary com token2idx.
    matrix        : matriz pré-inicializada com OOV
                    aleatórios.
    embedding_dim : dimensão esperada dos vetores.
    Retorna
    -------
    torch.Tensor de shape (vocab_size, embedding_dim).
    Retorna tensor com embeddings aleatórios em caso
    de falha na leitura.
    """
    logger.info(
        "Carregando FastText .vec: %s", path
    )
    found = 0
    try:
        with open(
            path, "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            # Detectar header: "N_WORDS DIM"
            first_line = f.readline()
            parts      = first_line.strip().split(" ")
            is_header  = (
                len(parts) == 2
                and parts[0].isdigit()
                and parts[1].isdigit()
            )
            if not is_header:
                # Primeira linha é um vetor
                token = parts[0]
                if token in vocab.token2idx:
                    try:
                        vec = np.array(
                            parts[1:],
                            dtype=np.float32,
                        )
                        if len(vec) == embedding_dim:
                            matrix[
                                vocab.token2idx[token]
                            ] = vec
                            found += 1
                    except ValueError:
                        pass
            # Processar restante do arquivo
            for line in f:
                parts = line.rstrip().split(" ")
                if not parts:
                    continue
                token = parts[0]
                if token not in vocab.token2idx:
                    continue
                try:
                    vec = np.array(
                        parts[1:], dtype=np.float32
                    )
                    if len(vec) == embedding_dim:
                        matrix[
                            vocab.token2idx[token]
                        ] = vec
                        found += 1
                except ValueError:
                    continue
                n_tokens = len(vocab) - 2
        coverage = (
            100.0 * found / n_tokens
            if n_tokens > 0
            else 0.0
        )
        # UNK = média dos vetores carregados
        # (consistente com _load_bin)
        if found > 0:
            # índices 0=PAD, 1=UNK; carregados a partir de 2
            loaded_vecs = matrix[2:]
            non_random = np.linalg.norm(
                loaded_vecs, axis=1
            ) > 0
            if non_random.any():
                matrix[UNK_IDX] = (
                    loaded_vecs[non_random].mean(axis=0)
                )
                logger.debug(
                    "_load_vec: UNK inicializado "
                    "como média de %d vetores.",
                    int(non_random.sum()),
                )
        logger.info(
            "FastText .vec: %d/%d tokens (%.1f%%)",
            found, n_tokens, coverage,
        )
        if coverage < 10.0:
            logger.warning(
                "FastText cobertura baixa (%.1f%%). "
                "Verificar idioma do .vec.",
                coverage,
            )
        return torch.tensor(matrix)
    except Exception as e:
        logger.error(
            "_load_vec falhou: %s — "
            "usando embeddings aleatórios.", e,
        )
        return torch.tensor(matrix)
# ---------------------------------------------------------------------------
# Collate para DataLoader
# ---------------------------------------------------------------------------
def collate_batch(
    batch: List[Tuple[List[int], int]],
    pad_idx: int = PAD_IDX,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Agrupa lista de (token_ids, label) em tensors
    padded para uso no DataLoader.
    Parâmetros
    ----------
    batch   : lista de (token_ids, label).
    pad_idx : índice de padding (padrão: PAD_IDX=0).
    Retorna
    -------
    (padded_sequences, lengths, labels)
      padded_sequences : (batch_size, max_len)
      lengths          : (batch_size,)
      labels           : (batch_size,)
    Raises
    ------
    ValueError
        Se batch estiver vazio ou com formato inválido.
    """
    if not batch:
        raise ValueError(
            "collate_batch recebeu batch vazio. "
            "Verifique o DataLoader e o dataset."
        )
    # CORRIGIDO: verificar formato antes de unpack
    for i, item in enumerate(batch):
        if not isinstance(item, (tuple, list)):
            raise ValueError(
                f"collate_batch: item {i} não é "
                f"tuple/list. "
                f"Tipo recebido: {type(item).__name__}"
            )
        if len(item) != 2:
            raise ValueError(
                f"collate_batch: item {i} tem "
                f"{len(item)} elemento(s), "
                f"esperado 2 (token_ids, label)."
            )
    token_ids_list, labels = zip(*batch)
    # Conversão explícita para int — protege contra
    # labels como float, np.int64 ou string.
    labels_int = [int(lbl) for lbl in labels]
    lengths = torch.tensor(
        [len(t) for t in token_ids_list],
        dtype=torch.long,
    )
    max_len = max(int(lengths.max().item()), 1)
    padded  = torch.full(
        (len(batch), max_len),
        pad_idx,
        dtype=torch.long,
    )
    for i, ids in enumerate(token_ids_list):
        if ids:
            padded[i, : len(ids)] = torch.tensor(
                ids, dtype=torch.long
            )
    return (
        padded,
        lengths,
        torch.tensor(labels_int, dtype=torch.long),
    )