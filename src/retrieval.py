"""
Module Retrieval dan Ranking untuk Information Retrieval (IR) System.
Mengintegrasikan:
1. Whoosh Inverted Index untuk pencarian kandidat dokumen
2. CountVectorizer (Bag of Words / BoW) untuk representasi vektor dokumen & query
3. Cosine Similarity untuk ranking dokumen berdasarkan kemiripan vektor
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from whoosh.index import open_dir
from whoosh.qparser import QueryParser, MultifieldParser, OrGroup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text
from src.indexer import DEFAULT_INDEX_DIR, is_index_built


class SearchResult:
    """Struktur data untuk menyimpan hasil pencarian dokumen."""
    def __init__(
        self,
        rank: int,
        score: float,
        doc_id: str,
        source: str,
        judul: str,
        konten: str,
        snippet: str
    ):
        self.rank = rank
        self.score = score
        self.doc_id = doc_id
        self.source = source
        self.judul = judul
        self.konten = konten
        self.snippet = snippet

    def to_dict(self) -> Dict:
        return {
            'rank': self.rank,
            'score': self.score,
            'doc_id': self.doc_id,
            'source': self.source,
            'judul': self.judul,
            'konten': self.konten,
            'snippet': self.snippet
        }


def extract_snippet(text: str, query_tokens: List[str], max_len: int = 240) -> str:
    """
    Mengekstrak cuplikan (snippet) teks yang paling relevan
    berdasarkan kemunculan token query.
    """
    if not text:
        return ""

    text = " ".join(text.split())
    if len(text) <= max_len:
        return text

    # Cari posisi kemunculan token query pertama
    lower_text = text.lower()
    first_pos = -1
    for token in query_tokens:
        pos = lower_text.find(token)
        if pos != -1 and (first_pos == -1 or pos < first_pos):
            first_pos = pos

    if first_pos == -1 or first_pos < 50:
        start = 0
    else:
        start = max(0, first_pos - 40)

    end = min(len(text), start + max_len)
    snippet = text[start:end]

    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(text) else ""
    return prefix + snippet.strip() + suffix


class InformationRetrievalEngine:
    """
    Mesin Penelusuran Informasi (IR Engine).
    Menggabungkan Whoosh Search + Bag of Words (CountVectorizer) + Cosine Similarity.
    """
    def __init__(self, index_dir: str = DEFAULT_INDEX_DIR):
        self.index_dir = index_dir
        self._index = None

    def _get_index(self):
        """Membuka indeks Whoosh jika tersedia."""
        if not is_index_built(self.index_dir):
            raise FileNotFoundError(
                f"Indeks belum dibangun di '{self.index_dir}'. "
                "Silakan jalankan menu [1] Load & Index Dataset terlebih dahulu."
            )
        if self._index is None:
            self._index = open_dir(self.index_dir)
        return self._index

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_limit: int = 200,
        source_filter: Optional[str] = None,
        use_stemming: bool = False
    ) -> Tuple[List[SearchResult], Dict]:
        """
        Melakukan pencarian query dan ranking dokumen:
        1. Preprocessing Query (Case folding, Tokenizing, Stopword removal)
        2. Whoosh Inverted Index Query (Mengambil kandidat dokumen relevan)
        3. BoW Representasi via CountVectorizer
        4. Cosine Similarity Calculation
        5. Pemeringkatan (Ranking) Top-K

        Returns:
            Tuple (daftar SearchResult, dictionary metadata proses IR)
        """
        ix = self._get_index()

        # 1. Preprocessing Query
        clean_query, query_tokens = preprocess_text(query, use_stemming=use_stemming)

        metadata = {
            'original_query': query,
            'clean_query': clean_query,
            'query_tokens': query_tokens,
            'num_candidates_retrieved': 0,
            'vocabulary_size': 0,
            'execution_status': 'success'
        }

        if not clean_query.strip():
            metadata['execution_status'] = 'empty_query'
            return [], metadata

        # 2. Whoosh Search (Kandidat Retrieval)
        candidate_docs = []
        candidate_clean_texts = []

        with ix.searcher() as searcher:
            # Menggunakan OrGroup agar dokumen yang mengandung salah satu kata query dapat diambil
            parser = MultifieldParser(["clean_text", "judul"], schema=ix.schema, group=OrGroup.factory(0.9))
            parsed_q = parser.parse(clean_query)

            # Filter source jika ditentukan
            filter_query = None
            if source_filter:
                from whoosh.query import Term
                filter_query = Term("source", source_filter)

            results = searcher.search(parsed_q, limit=candidate_limit, filter=filter_query)
            metadata['num_candidates_retrieved'] = len(results)

            if len(results) == 0:
                metadata['execution_status'] = 'no_candidates_found'
                return [], metadata

            for hit in results:
                candidate_docs.append({
                    'doc_id': hit['doc_id'],
                    'source': hit['source'],
                    'judul': hit['judul'],
                    'konten': hit['konten'],
                    'clean_text': hit['clean_text']
                })
                candidate_clean_texts.append(hit['clean_text'])

        # 3. Bag-of-Words (BoW) Representasi via CountVectorizer
        vectorizer = CountVectorizer()
        corpus = [clean_query] + candidate_clean_texts
        dtm = vectorizer.fit_transform(corpus)

        metadata['vocabulary_size'] = len(vectorizer.vocabulary_)
        
        # 4. Cosine Similarity Calculation
        query_vector = dtm[0]           # Vektor Baris 0 = Query
        document_vectors = dtm[1:]      # Vektor Baris 1..N = Dokumen-dokumen kandidat

        cos_scores = cosine_similarity(query_vector, document_vectors).flatten()

        # 5. Ranking berdasarkan skor kemiripan tertinggi (Descending)
        sorted_indices = cos_scores.argsort()[::-1]

        final_results = []
        for rank, doc_idx in enumerate(sorted_indices[:top_k], start=1):
            score = float(cos_scores[doc_idx])
            doc = candidate_docs[doc_idx]
            snippet = extract_snippet(doc['konten'], query_tokens)

            final_results.append(
                SearchResult(
                    rank=rank,
                    score=score,
                    doc_id=doc['doc_id'],
                    source=doc['source'],
                    judul=doc['judul'],
                    konten=doc['konten'],
                    snippet=snippet
                )
            )

        return final_results, metadata
