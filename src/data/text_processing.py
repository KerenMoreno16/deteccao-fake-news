"""
text_processing.py
Each function takes a string and returns a string so they can be
composed freely. The `clean_text` function applies the full pipeline
in the correct order.
"""
import html
import re
import unicodedata
from typing import Optional
# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import time)
# ---------------------------------------------------------------------------
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_MENTION_RE    = re.compile(r"@\w+")
_HASHTAG_RE    = re.compile(r"#(\w+)")
_HASHTAG_REMOVE_RE = re.compile(r"#\w+")
_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000027BF"
    "\U0001F900-\U0001FAFF"
    "\U0000FE0F\U0000200D]+",
    flags=re.UNICODE,
)
_WHITESPACE_RE       = re.compile(r"\s+")
_REPEATED_PUNCT_RE   = re.compile(r"([!?.]){2,}")
# CORRIGIDO: padrão explícito sem \w para evitar
# inconsistência entre sistemas operacionais.
# Remove tudo que não seja letra (ASCII + acentuadas
# comuns em PT/ES), dígito ou espaço.
# Underscore é removido explicitamente.
_ALPHANUM_RE = re.compile(
    r"[^a-záéíóúâêîôûãõäëïöüçàèìòùñ"
    r"ÁÉÍÓÚÂÊÎÔÛÃÕÄËÏÖÜÇÀÈÌÒÙÑ"
    r"0-9\s]",
    flags=re.UNICODE,
)
def decode_html_entities(text: str) -> str:
    """Decode HTML entities such as &amp; &lt; &gt;."""
    return html.unescape(text)
def remove_urls(
    text: str,
    placeholder: str = " ",
) -> str:
    """Remove all URLs (http, https, www)."""
    return _URL_RE.sub(placeholder, text)
def remove_mentions(
    text: str,
    placeholder: str = " ",
) -> str:
    """Remove @user mentions."""
    return _MENTION_RE.sub(placeholder, text)
def normalize_hashtags(text: str) -> str:
    """Replace '#word' with 'word'."""
    return _HASHTAG_RE.sub(r"\1", text)
def remove_hashtags(
    text: str,
    placeholder: str = " ",
) -> str:
    """Remove entire hashtag tokens."""
    return _HASHTAG_REMOVE_RE.sub(placeholder, text)
def remove_emojis(
    text: str,
    placeholder: str = " ",
) -> str:
    """Remove all Unicode emoji characters."""
    return _EMOJI_RE.sub(placeholder, text)
def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC and remove non-ASCII
    control characters, preserving \\n, \\t and
    ZWJ (U+200D, used in composed emoji sequences).
    """
    text = unicodedata.normalize("NFC", text)
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("C")
        or ch in "\n\t\u200d"
    )
def collapse_whitespace(text: str) -> str:
    """
    Replace runs of whitespace (including newlines)
    with a single space.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()
def normalize_repeated_punctuation(text: str) -> str:
    """Reduce '!!!' or '???' to a single character."""
    return _REPEATED_PUNCT_RE.sub(r"\1", text)
def truncate_tokens(
    text: str,
    max_tokens: int,
) -> str:
    """
    Keep only the first *max_tokens*
    whitespace-separated tokens.
    """
    tokens = text.split()
    return " ".join(tokens[:max_tokens])
def to_lowercase(text: str) -> str:
    return text.lower()

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def clean_text(
    text: str,
    lowercase: bool = True,
    remove_url: bool = True,
    remove_mention: bool = True,
    hashtag_mode: str = "normalize",
    emoji_mode: str = "keep",
    decode_html: bool = True,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Apply the full cleaning pipeline.
    Parameters
    ----------
    text         : input string (non-string inputs are
                   coerced to str).
    lowercase    : convert to lowercase.
    remove_url   : remove http/https/www URLs.
    remove_mention: remove @user mentions.
    hashtag_mode : "normalize" keeps the word (removes #),
                   "remove" drops the whole token,
                   "keep" leaves the hashtag as-is.
    emoji_mode   : "keep" preserves emojis;
                   "remove" strips them.
    decode_html  : decode HTML entities before cleaning.
    max_tokens   : if set, truncate to first N tokens
                   after all other processing.
    Returns
    -------
    Cleaned string. Never returns None.
    Empty string if input is empty after cleaning.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    if not text.strip():
        return ""
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
    # else: keep as-is
    if emoji_mode == "remove":
        text = remove_emojis(text)
    text = normalize_repeated_punctuation(text)
    text = collapse_whitespace(text)
    if lowercase:
        text = to_lowercase(text)
    if max_tokens is not None:
        text = truncate_tokens(text, max_tokens)
    return text