"""
Module Preprocessing Teks untuk Information Retrieval (IR) System
Mendukung:
1. Case folding
2. Tokenization & pembersihan karakter khusus
3. Stopword removal (PySastrawi + extended Indonesian stopwords)
4. Stemming (PySastrawi Nazief & Adriani) - Opsional / Toggleable
"""

import re
import string
import unicodedata
from typing import List, Tuple, Set

# Inisialisasi Stopwords Bahasa Indonesia
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _sw_factory = StopWordRemoverFactory()
    INDONESIAN_STOPWORDS: Set[str] = set(_sw_factory.get_stop_words())
except ImportError:
    # Fallback minimal jika Sastrawi belum terinstall
    INDONESIAN_STOPWORDS: Set[str] = {
        'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua',
        'ia', 'seperti', 'jika', 'sehingga', 'kembali', 'dan', 'ini', 'karena', 'kepada',
        'oleh', 'saat', 'harus', 'sementara', 'setelah', 'belum', 'kami', 'sekitar', 'bagi',
        'serta', 'di', 'dari', 'tersebut', 'dalam', 'bisa', 'atau', 'hanya', 'ada', 'itu'
    }

# Tambahan stopword umum yang sering muncul di artikel web / jurnal
EXTRA_STOPWORDS: Set[str] = {
    'yg', 'dgn', 'dr', 'kpd', 'utk', 'sdh', 'tdk', 'bkn', 'dlm', 'dg', 'sbg',
    'hlm', 'hal', 'vol', 'no', 'halaman', 'issn', 'isbn', 'doi', 'com', 'co', 'id'
}
INDONESIAN_STOPWORDS.update(EXTRA_STOPWORDS)

# Regex pattern untuk tokenisasi kata (hanya huruf alfabet, minimal 2 karakter)
TOKEN_PATTERN = re.compile(r'\b[a-zA-Z]{2,}\b')

# Lazy loaded stemmer
_stemmer_instance = None

def get_stemmer():
    """Mengambil instance stemmer PySastrawi (lazy loading)."""
    global _stemmer_instance
    if _stemmer_instance is None:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            factory = StemmerFactory()
            _stemmer_instance = factory.create_stemmer()
        except ImportError:
            _stemmer_instance = None
    return _stemmer_instance


def case_folding(text: str) -> str:
    """
    Mengubah semua karakter menjadi huruf kecil (lowercase)
    dan menormalkan karakter unicode.
    """
    if not isinstance(text, str):
        text = str(text or "")
    text = unicodedata.normalize('NFKD', text)
    return text.lower().strip()


def tokenize(text: str) -> List[str]:
    """
    Memecah teks menjadi daftar token kata alfabetik.
    Menghilangkan tanda baca, angka, dan karakter simbol.
    """
    return TOKEN_PATTERN.findall(text)


def remove_stopwords(tokens: List[str], custom_stopwords: Set[str] = None) -> List[str]:
    """
    Menghapus stopword bahasa Indonesia dari daftar token.
    """
    sw = custom_stopwords if custom_stopwords is not None else INDONESIAN_STOPWORDS
    return [token for token in tokens if token not in sw]


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Melakukan stemming pada daftar token menggunakan algoritma Nazief & Adriani (PySastrawi).
    """
    stemmer = get_stemmer()
    if stemmer is None:
        return tokens
    # PySastrawi stemmer lebih cepat memproses gabungan kalimat
    sentence = " ".join(tokens)
    stemmed_sentence = stemmer.stem(sentence)
    return stemmed_sentence.split()


def preprocess_text(text: str, use_stemming: bool = False) -> Tuple[str, List[str]]:
    """
    Pipeline lengkap preprocessing teks:
    1. Case Folding
    2. Tokenization
    3. Stopword Removal
    4. Stemming (opsional)

    Returns:
        Tuple (clean_text_string, list_of_clean_tokens)
    """
    # 1. Case Folding
    folded = case_folding(text)

    # 2. Tokenize
    tokens = tokenize(folded)

    # 3. Stopword Removal
    filtered_tokens = remove_stopwords(tokens)

    # 4. Stemming (opsional)
    if use_stemming:
        final_tokens = stem_tokens(filtered_tokens)
    else:
        final_tokens = filtered_tokens

    clean_string = " ".join(final_tokens)
    return clean_string, final_tokens
