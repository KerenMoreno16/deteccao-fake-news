# -*- coding: utf-8 -*-
"""
constants.py — Constantes globais compartilhadas
entre todos os módulos.
Centraliza PT_DATASETS, EN_DATASETS,
TEMPORAL_DATASETS e DATASET_LANGUAGE para evitar
duplicação e inconsistências entre arquivos.
Correções aplicadas:
  - FASTTEXT_MODEL_BY_LANGUAGE corrigido para usar
    data/embeddings em vez de embeddings.
  - Comentário sobre .bin/.vec corrigido:
    .bin é suportado via _load_bin() em embeddings.py.
  - assert substituído por raise ValueError para
    verificação robusta em tempo de importação.
"""
from pathlib import Path
from typing import Dict, Set
__all__ = [
    "PT_DATASETS",
    "EN_DATASETS",
    "TEMPORAL_DATASETS",
    "DATASET_LANGUAGE",
    "BERT_MODEL_BY_LANGUAGE",
    "FASTTEXT_MODEL_BY_LANGUAGE",
]
_PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)
# ---------------------------------------------------------------------------
# Grupos de datasets por idioma
# ---------------------------------------------------------------------------
PT_DATASETS: Set[str] = {
    "faketruebr",
    "fakebr",
    "fakerecogna",
}
EN_DATASETS: Set[str] = {
    "fakenewsnet",
    "pheme",
    "constraint_covid19",
}
# ---------------------------------------------------------------------------
# Datasets com timestamp disponível (Experimento D)
# ---------------------------------------------------------------------------
TEMPORAL_DATASETS: Set[str] = {
    "fakerecogna",
    "pheme",
}
# Verificação de consistência em tempo de importação.
# Usa raise ValueError em vez de assert para garantir
# que a verificação não seja desabilitada com
# python -O (modo otimizado).
_invalid = TEMPORAL_DATASETS - (
    PT_DATASETS | EN_DATASETS
)
if _invalid:
    raise ValueError(
        f"TEMPORAL_DATASETS contém datasets não "
        f"registrados em PT_DATASETS | EN_DATASETS: "
        f"{_invalid}"
    )
del _invalid
# ---------------------------------------------------------------------------
# Mapeamento dataset -> idioma
# ---------------------------------------------------------------------------
DATASET_LANGUAGE: Dict[str, str] = {
    **{ds: "pt" for ds in PT_DATASETS},
    **{ds: "en" for ds in EN_DATASETS},
}
# ---------------------------------------------------------------------------
# Modelos Transformer por idioma
# ---------------------------------------------------------------------------
BERT_MODEL_BY_LANGUAGE: Dict[str, str] = {
    "pt": "neuralmind/bert-base-portuguese-cased",
    "en": "bert-base-uncased",
}
# ---------------------------------------------------------------------------
# Caminhos padrão para embeddings FastText
#
# Caminho usa data/embeddings (convenção do projeto).
#
# Formato .bin preferido:
#   - Cobertura ~100% via decomposição em subwords
#   - Requer: pip install fasttext-wheel
#   - Suportado por _load_bin() em embeddings.py
#
# Fallback .vec (renomear extensão se necessário):
#   - Cobertura ~40-60% (apenas tokens exatos)
#   - Sem dependências extras
#   - Suportado por _load_vec() em embeddings.py
#
# Download:
#   PT .bin: https://dl.fbaipublicfiles.com/fasttext/
#            vectors-crawl/cc.pt.300.bin.gz
#   EN .bin: https://dl.fbaipublicfiles.com/fasttext/
#            vectors-crawl/cc.en.300.bin.gz
# ---------------------------------------------------------------------------
FASTTEXT_MODEL_BY_LANGUAGE: Dict[str, str] = {
    "pt": str(
        _PROJECT_ROOT
        / "data"
        / "embeddings"
        / "cc.pt.300.bin"
    ),
    "en": str(
        _PROJECT_ROOT
        / "data"
        / "embeddings"
        / "cc.en.300.bin"
    ),
}