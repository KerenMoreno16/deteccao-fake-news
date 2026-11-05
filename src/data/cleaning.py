"""
cleaning.py — Pipeline modular de higienização textual.
"""
import html
import re
import unicodedata
from typing import List, Optional

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_HASHTAG_REMOVE_RE = re.compile(r"#\w+")
# Faixas específicas de emoji/pictogramas (a faixa anterior U+24C2–U+1F251
# capturava letras circuladas e símbolos que NÃO são emoji).
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001F900-\U0001FAFF"
    "\U0000FE0F\U0000200D]+",
    flags=re.UNICODE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.]){2,}")
# IGNORECASE: preserva também A-Z sem acento (textos cased do BERTimbau).
_ALPHANUM_RE = re.compile(
    r"[^a-záéíóúâêîôûãõäëïöüçàèìòùñ0-9\s]",
    flags=re.UNICODE | re.IGNORECASE,
)

def decode_html_entities(text: str) -> str:
    return html.unescape(text)


def remove_urls(text: str, placeholder: str = " ") -> str:
    return _URL_RE.sub(placeholder, text)


def remove_mentions(text: str, placeholder: str = " ") -> str:
    return _MENTION_RE.sub(placeholder, text)


def normalize_hashtags(text: str) -> str:
    return _HASHTAG_RE.sub(r"\1", text)


def remove_hashtags(text: str, placeholder: str = " ") -> str:
    return _HASHTAG_REMOVE_RE.sub(placeholder, text)


def remove_emojis(text: str, placeholder: str = " ") -> str:
    return _EMOJI_RE.sub(placeholder, text)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # Mantém \n, \t e o ZWJ (U+200D).
    # Nota: se remove_emojis() for aplicado antes,
    # ZWJs órfãos podem permanecer — inofensivos para
    # tokenizadores mas visíveis em inspeção raw.
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("C") or ch in "\n\t\u200d"
    )


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_repeated_punctuation(text: str) -> str:
    return _REPEATED_PUNCT_RE.sub(r"\1", text)


def truncate_tokens(text: str, max_tokens: int) -> str:
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def to_lowercase(text: str) -> str:
    return text.lower()

# Modelos Transformer: o próprio tokenizer decide a caixa
# (BERTimbau é CASED → preservar maiúsculas; bert-base-uncased já faz lowercase).
TRANSFORMER_MODEL_TYPES = frozenset({
    "bertimbau", "bert",
    "neuralmind/bert-base-portuguese-cased", "bert-base-uncased",
})


def apply_model_casing(texts, model_name: Optional[str], lowercase: bool = True) -> List[str]:
    """
    Aplica a política de caixa POR MODELO sobre textos já higienizados (cased).
    TF-IDF/LR/MLP/BiLSTM: minúsculas (se `lowercase`); Transformers: intactos.
    """
    out = [t if isinstance(t, str) else str(t) for t in texts]
    if lowercase and model_name not in TRANSFORMER_MODEL_TYPES:
        out = [t.lower() for t in out]
    return out


def clean_text(
    text: str,
    lowercase: bool = True,
    model_type: Optional[str] = None,
    remove_url: bool = True,
    remove_mention: bool = True,
    hashtag_mode: str = "normalize",
    emoji_mode: str = "keep",
    decode_html: bool = True,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Higienização do texto preservando maiúsculas para modelos Cased (ex: BERTimbau).
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    if not text.strip():
        return ""

    # Transformers decidem a caixa no tokenizer (BERTimbau é cased: preservar
    # maiúsculas de entidades/siglas; bert-base-uncased faz lowercase sozinho).
    if model_type in TRANSFORMER_MODEL_TYPES:
        lowercase = False

    if decode_html:
        text = decode_html_entities(text)
    text = normalize_unicode(text)
    if remove_url:
        text = remove_urls(text)
    if remove_mention:
        text = remove_mentions(text)
    if hashtag_mode == "normalize":
        text = normalize_hashtags(text)
    elif hashtag_mode == "remove":
        text = remove_hashtags(text)
    if emoji_mode == "remove":
        text = remove_emojis(text)
    text = normalize_repeated_punctuation(text)
    text = collapse_whitespace(text)
    if lowercase:
        text = to_lowercase(text)
    if max_tokens is not None:
        text = truncate_tokens(text, max_tokens)
    return text