import os
import time
import pandas as pd
from typing import Dict, List, Optional, Callable
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StandardAnalyzer

from src.preprocessing import preprocess_text

# Definisi nama dataset dan file CSV terkait
DATASET_FILES = {
    'etd-usk': 'etd_usk.csv',
    'etd-ugm': 'etd_ugm.csv',
    'kompas': 'kompas.csv',
    'tempo': 'tempo.csv',
    'mojok': 'mojok.csv',
}

DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'indexdir')
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')


def get_schema() -> Schema:
    return Schema(
        doc_id=ID(stored=True, unique=True),
        source=ID(stored=True),
        judul=TEXT(stored=True),
        konten=TEXT(stored=True),
        clean_text=TEXT(stored=True, analyzer=StandardAnalyzer())
    )


def is_index_built(index_dir: str = DEFAULT_INDEX_DIR) -> bool:
    if not os.path.exists(index_dir):
        return False
    if not exists_in(index_dir):
        return False
    try:
        ix = open_dir(index_dir)
        with ix.searcher() as searcher:
            count = searcher.doc_count()
            return count > 0
    except Exception:
        return False


def get_index_stats(index_dir: str = DEFAULT_INDEX_DIR) -> Dict:
    """Mengambil statistik dokumen yang tersimpan di indeks Whoosh."""
    if not is_index_built(index_dir):
        return {'status': 'empty', 'total_docs': 0, 'sources': {}}

    ix = open_dir(index_dir)
    stats = {'status': 'ready', 'total_docs': 0, 'sources': {}}
    with ix.searcher() as searcher:
        stats['total_docs'] = searcher.doc_count()
        # Hitung per source
        for source_name in DATASET_FILES.keys():
            count = searcher.doc_count_filtered(searcher.reader(), ix.schema['source'].to_bytes(source_name)) if hasattr(searcher, 'doc_count_filtered') else 0
            stats['sources'][source_name] = count
    return stats


def get_dataset_inventory(dataset_dir: str = DATASET_DIR) -> Dict[str, Dict]:
    """Mengambil informasi ukuran file dan jumlah baris masing-masing dataset."""
    info = {}
    for key, filename in DATASET_FILES.items():
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            info[key] = {
                'filename': filename,
                'path': filepath,
                'size_mb': round(size_mb, 2),
                'available': True
            }
        else:
            info[key] = {
                'filename': filename,
                'path': filepath,
                'size_mb': 0,
                'available': False
            }
    return info


def build_index(
    sources: Optional[List[str]] = None,
    limit_per_source: Optional[int] = None,
    use_stemming: bool = False,
    index_dir: str = DEFAULT_INDEX_DIR,
    dataset_dir: str = DATASET_DIR,
    progress_callback: Optional[Callable[[str, int, int, float], None]] = None
) -> int:
    """
    Membangun indeks Whoosh dari file CSV dataset.

    Parameters:
        sources: List nama dataset yang ingin diindeks (default: semua 5 dataset)
        limit_per_source: Batas dokumen per file CSV (None = seluruh dokumen)
        use_stemming: Apakah menerapkan stemming (default: False untuk efisiensi)
        index_dir: Direktori penyimpanan indeks Whoosh
        dataset_dir: Direktori tempat file CSV dataset berada
        progress_callback: Fungsi callback untuk memantau progress (source, current, total, elapsed)

    Returns:
        Jumlah total dokumen yang berhasil diindeks.
    """
    if sources is None:
        sources = list(DATASET_FILES.keys())

    os.makedirs(index_dir, exist_ok=True)
    schema = get_schema()
    ix = create_in(index_dir, schema)
    writer = ix.writer(limitmb=512, procs=1)

    total_indexed = 0
    start_time = time.time()

    for source in sources:
        if source not in DATASET_FILES:
            continue
        filename = DATASET_FILES[source]
        filepath = os.path.join(dataset_dir, filename)

        if not os.path.exists(filepath):
            print(f"[Warning] File {filepath} tidak ditemukan!")
            continue

        # Baca CSV dengan pandas
        df = pd.read_csv(filepath, nrows=limit_per_source)
        # Tangani kemungkinan missing values
        df['judul'] = df['judul'].fillna('').astype(str)
        df['konten'] = df['konten'].fillna('').astype(str)

        total_rows = len(df)

        for idx, row in df.iterrows():
            judul = row['judul']
            konten = row['konten']
            doc_id = f"{source}_{idx}"

            # Preprocessing judul dan konten
            raw_text = f"{judul} {konten}"
            clean_text, _ = preprocess_text(raw_text, use_stemming=use_stemming)

            writer.add_document(
                doc_id=doc_id,
                source=source,
                judul=judul,
                konten=konten,
                clean_text=clean_text
            )
            total_indexed += 1

            if progress_callback is not None and ((idx + 1) % 250 == 0 or (idx + 1) == total_rows):
                elapsed = time.time() - start_time
                progress_callback(source, idx + 1, total_rows, elapsed)

    # Commit indeks ke disk
    writer.commit()
    return total_indexed
