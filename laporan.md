# LAPORAN TUGAS SISTEM PENELUSURAN INFORMASI (INFORMATION RETRIEVAL)
## Implementasi Mesin Penelusuran Lintas Dataset dengan Whoosh Inverted Index, Bag of Words, dan Cosine Similarity

---

* **Mata Kuliah** : Penelusuran Informasi (Information Retrieval)
* **Topik** : Perancangan dan Implementasi Sistem Information Retrieval Berbasis CLI Interaktif
* **Dataset yang Digunakan** : 5 Sumber Teks Berbahasa Indonesia (ETD USK, ETD UGM, Kompas, Tempo, Mojok.co)
* **Teknologi Utama** : Python 3.9+, Whoosh Search Engine, Scikit-Learn (CountVectorizer, Cosine Similarity), PySastrawi

---

## 2. Pendahuluan

### 2.1 Latar Belakang
Pertumbuhan data tekstual digital dalam bahasa Indonesia berkembang sangat pesat dalam satu dekade terakhir. Sumber-sumber informasi ini tersebar di berbagai domain, mulai dari repositori akademik perguruan tinggi berupa karya akhir tesis dan disertasi (*Electronic Theses and Dissertations* / ETD), portal berita media massa arus utama (*mainstream media*), hingga situs web opini populer dan esai kultural. Dalam lingkup repositori yang mencapai puluhan ribu dokumen, menemukan informasi yang relevan dengan kebutuhan pengguna (*information need*) menjadi tantangan besar.

Pencarian tradisional yang hanya mengandalkan pencocokan string secara harfiah (*exact substring match* seperti perintah `grep` atau query basis data relasional `LIKE '%keyword%'`) memiliki kelemahan mendasar:
1. **Tidak Memiliki Skoring Relevansi**: Pencarian eksak hanya menghasilkan keluaran biner (cocok atau tidak cocok), tanpa mampu mengukur seberapa dekat keterkaitan dokumen tersebut dengan topik yang dicari pengguna.
2. **Keterbatasan Skalabilitas**: Melakukan pemindaian linier (*linear scan*) pada puluhan ribu dokumen teks lengkap membutuhkan waktu komputasi yang tidak realistis untuk aplikasi waktu nyata (*real-time*).
3. **Ketidakmampuan Mengolah Teks Alami**: Variasi penulisan huruf besar-kecil, imbuhan kata, tanda baca, serta kata sambung yang tidak bernilai informasi (seperti *yang, dan, di, dari*) dapat mengacaukan ketepatan hasil pencarian.

Oleh karena itu, diperlukan sistem **Information Retrieval (IR)** berbasis ruang vektor (*Vector Space Model*) yang dikombinasikan dengan struktur indeks terbalik (*Inverted Index*). Dengan arsitektur ini, sistem mampu menyaring dokumen kandidat secara instan dan menghitung bobot kemiripan semantik berbasis frekuensi kata (*Bag of Words*) melalui metrik **Cosine Similarity**.

### 2.2 Rumusan Masalah
Berdasarkan latar belakang tersebut, perancangan sistem ini berfokus pada:
1. Bagaimana mengintegrasikan 5 dataset heterogen berbahasa Indonesia (akademik, berita formal, dan artikel santai/satir) ke dalam satu repositori dokumen terstruktur?
2. Bagaimana membangun pipeline pemrosesan teks (*text preprocessing*) berbahasa Indonesia yang efisien untuk membersihkan kebisingan (*noise*) tanpa menghilangkan esensi informasi?
3. Bagaimana merancang arsitektur dua tahap (*two-stage retrieval*) yang memadukan kecepatan penelusuran indeks Whoosh dengan ketelitian perankingan *Cosine Similarity* berbasis representasi *Bag of Words*?
4. Bagaimana menyajikan antarmuka baris perintah (*Command-Line Interface* / CLI) yang informatif, interaktif, dan mudah dioperasikan oleh pengguna umum?

### 2.3 Tujuan Proyek
Tujuan yang dicapai dalam penyusunan proyek ini antara lain:
1. Membangun sistem Information Retrieval yang modular, terstruktur rapi, dan mudah dikembangkan lebih lanjut.
2. Mengimplementasikan indexing berkecepatan tinggi menggunakan pustaka Whoosh untuk mengindeks koleksi hingga ~49.484 dokumen teks berbahasa Indonesia.
3. Menerapkan representasi vektor dokumen dan query melalui model *Bag of Words* (BoW) menggunakan `CountVectorizer`.
4. Menghitung derajat kesamaan antara query pengguna dan dokumen kandidat menggunakan formula matematis *Cosine Similarity*, lalu menyajikan Top-5 hasil teratas secara terurut menurun (*descending*).
5. Menyediakan fitur navigasi CLI interaktif lengkap dengan penampil cuplikan teks (*snippet context extractor*) dan pembaca dokumen penuh (*full document reader*).

### 2.4 Cakupan dan Batasan Proyek
1. **Dataset**: Menggunakan 5 sumber dataset nyata dalam format CSV:
   - `etd-usk` : Tesis & disertasi Universitas Syiah Kuala (10.000 dokumen).
   - `etd-ugm` : Tesis & disertasi Universitas Gadjah Mada (10.000 dokumen).
   - `kompas` : Berita harian nasional Kompas (10.000 dokumen).
   - `tempo` : Berita investigasi dan opini Tempo (10.000 dokumen).
   - `mojok` : Esai populer dan kolom satir Mojok.co (9.484 dokumen).
   - Total korpus: **49.484 dokumen**.
2. **Bahasa**: Fokus pemrosesan pada teks berbahasa Indonesia, memanfaatkan daftar stopword Sastrawi yang diperluas dengan istilah populer media/web.
3. **Model Temu Kembali**: *Vector Space Model* berbasis kemunculan kata (*Bag of Words* Term Frequency) yang dirangking menggunakan *Cosine Similarity* terhadap kandidat yang disaring oleh *Whoosh*.

---

## 3. Desain Sistem dan Arsitektur

### 3.1 Arsitektur Sistem (Two-Stage Retrieval & Reranking)
Untuk mencapai keseimbangan optimal antara **kecepatan pencarian (latency)** dan **ketepatan pemeringkatan (ranking accuracy)**, sistem dirancang dengan pendekatan *Two-Stage IR Pipeline*:

```text
               +-----------------------------------+
               |       Input Query Pengguna        |
               +-----------------------------------+
                                 |
                                 v
               +-----------------------------------+
               |        Text Preprocessing         |
               | - Case Folding                    |
               | - Tokenization Alfabetik          |
               | - Indonesian Stopword Removal     |
               | - Stemming (Opsional/Toggleable)  |
               +-----------------------------------+
                                 |
                     Query Bersih + Token List
                                 |
                                 v
               +-----------------------------------+
               |  STAGE 1: Candidate Generation    |
               |  (Whoosh Inverted Index Search)   |
               |  - Skema: clean_text & judul      |
               |  - Operator: OrGroup coordination |
               |  - Mengambil N Kandidat (~200 dok)|
               +-----------------------------------+
                                 |
                  Sub-Korpus Dokumen Kandidat
                                 |
                                 v
               +-----------------------------------+
               |  STAGE 2: Vector Space & Scoring  |
               |  - Bag of Words (CountVectorizer) |
               |  - Matriks Term-Dokumen Dinamis   |
               |  - Hitung Cosine Similarity:      |
               |    cos(q, d) = (q . d) / (|q||d|) |
               +-----------------------------------+
                                 |
                                 v
               +-----------------------------------+
               |  Ranking & Snippet Extraction     |
               |  - Urutkan skor tertinggi (Top-5) |
               |  - Highlight konteks kemunculan   |
               +-----------------------------------+
                                 |
                                 v
               +-----------------------------------+
               |      Output UI Terminal CLI       |
               +-----------------------------------+
```

**Alasan Pemilihan Arsitektur Dua Tahap**:
Jika representasi matriks *Bag of Words* dan perkalian kosinus dihitung langsung ke seluruh 49.484 dokumen sekaligus, penggunaan memori RAM akan membengkak drastis dan respon pencarian menjadi lambat. Sebaliknya, jika hanya mengandalkan mesin pencari teks biasa, kita kehilangan fleksibilitas representasi ruang vektor murni. Kombinasi ini memberikan yang terbaik dari kedua dunia: Whoosh bertindak sebagai *filter pencarian berkecepatan tinggi* untuk menyaring ratusan kandidat berpotensi, lalu *CountVectorizer* dan *Cosine Similarity* melakukan pemeringkatan akhir yang presisi secara matematis.

### 3.2 Skema Indeks Dokumen (Whoosh Schema)
Whoosh menyimpan dokumen ke dalam struktur indeks terbalik (*Inverted Index*) pada direktori disk lokal (`indexdir/`). Skema yang didefinisikan mencakup 5 atribut utama:

| Field Name | Tipe Whoosh | Pengaturan | Deskripsi Fungsi |
|:-----------|:------------|:-----------|:-----------------|
| `doc_id` | `ID` | `stored=True, unique=True` | Identifikator unik dokumen, gabungan nama dataset dan nomor indeks (contoh: `kompas_104`, `etd-usk_89`). |
| `source` | `ID` | `stored=True` | Kategori dataset asal (`etd-usk`, `etd-ugm`, `kompas`, `tempo`, `mojok`), digunakan untuk fitur penyaringan sumber. |
| `judul` | `TEXT` | `stored=True` | Judul asli dokumen tanpa manipulasi untuk ditampilkan ke pengguna. |
| `konten` | `TEXT` | `stored=True` | Isi artikel atau abstrak lengkap dokumen asli, disimpan untuk pratinjau penuh. |
| `clean_text` | `TEXT` | `stored=True, analyzer=StandardAnalyzer()` | Teks gabungan judul dan isi yang telah melewati pembersihan (case folding, tanpa stopword) untuk pengindeksan kata kunci. |

### 3.3 Struktur Organisasi Direktori Proyek
Proyek dibangun dengan tata kelola kode yang modular demi memisahkan logika pemrosesan data, indexing, retrieval, dan antarmuka tampilan:

```text
UTS/
├── dataset/                    # Direktori penyimpanan 5 file CSV dataset
│   ├── etd_ugm.csv             # 10.000 dokumen tugas akhir UGM
│   ├── etd_usk.csv             # 10.000 dokumen tugas akhir USK
│   ├── kompas.csv              # 10.000 artikel berita Kompas
│   ├── mojok.csv               # 9.484 artikel esai/satir Mojok
│   └── tempo.csv               # 10.000 berita/opini majalah Tempo
├── indexdir/                   # Direktori penyimpanan file biner inverted index Whoosh
├── src/                        # Modul Python fungsional
│   ├── __init__.py             # Inisialisasi package
│   ├── preprocessing.py        # Modul pembersihan teks, tokenisasi, stopword, stemming
│   ├── indexer.py              # Modul definisi skema, pembacaan CSV, batch indexing
│   ├── retrieval.py            # Mesin penelusuran IR, ekstraksi BoW, dan Cosine Similarity
│   └── ui.py                   # Format tampilan CLI, warna ANSI terminal, visualisasi bar
├── main.py                     # Entry point interaktif menu aplikasi
├── requirements.txt            # Daftar pustaka eksternal yang dibutuhkan
├── README.md                   # Panduan instalasi dan dokumentasi umum
└── laporan.md                  # Laporan lengkap proyek
```

---

## 4. Implementasi (Kode & Penjelasan)

### 4.1 Modul Pemrosesan Teks (`src/preprocessing.py`)
Pemrosesan teks merupakan fondasi utama agar representasi kata dalam dokumen dan query berada dalam format standar yang seragam.

#### 1. Case Folding & Normalisasi Karakter
Karakter diubah seluruhnya menjadi huruf kecil (*lowercase*) dan dinormalisasi menggunakan dekomposisi Unicode NFKD agar karakter beraksen atau variasi simbol non-standar dapat dikonversi ke padanan karakter ASCII standarnya.

```python
def case_folding(text: str) -> str:
    """
    Mengubah semua karakter menjadi huruf kecil (lowercase)
    dan menormalkan karakter unicode.
    """
    if not isinstance(text, str):
        text = str(text or "")
    text = unicodedata.normalize('NFKD', text)
    return text.lower().strip()
```

#### 2. Tokenisasi Alfabetik
Pemisahan kata menggunakan regular expression `\b[a-zA-Z]{2,}\b` untuk menjamin hanya kata berbasis alfabet dengan panjang minimal 2 karakter yang diambil, secara otomatis membuang angka, tanda baca, simbol aneh, dan spasi berlebih.

```python
TOKEN_PATTERN = re.compile(r'\b[a-zA-Z]{2,}\b')

def tokenize(text: str) -> List[str]:
    """
    Memecah teks menjadi daftar token kata alfabetik.
    Menghilangkan tanda baca, angka, dan karakter simbol.
    """
    return TOKEN_PATTERN.findall(text)
```

#### 3. Penghapusan Stopword (Stopword Removal)
Sistem menggunakan kamus stopword bawaan dari pustaka PySastrawi (berisi ratusan kata fungsional bahasa Indonesia), yang diperkaya dengan set kata stopword kontekstual umum di lingkungan artikel web dan karya ilmiah (seperti *yg, dgn, hlm, hal, vol, doi, com, id*).

```python
# Tambahan stopword umum di artikel web dan jurnal akademik
EXTRA_STOPWORDS: Set[str] = {
    'yg', 'dgn', 'dr', 'kpd', 'utk', 'sdh', 'tdk', 'bkn', 'dlm', 'dg', 'sbg',
    'hlm', 'hal', 'vol', 'no', 'halaman', 'issn', 'isbn', 'doi', 'com', 'co', 'id'
}
INDONESIAN_STOPWORDS.update(EXTRA_STOPWORDS)

def remove_stopwords(tokens: List[str], custom_stopwords: Set[str] = None) -> List[str]:
    sw = custom_stopwords if custom_stopwords is not None else INDONESIAN_STOPWORDS
    return [token for token in tokens if token not in sw]
```

#### 4. Stemming dan Pipeline Terintegrasi
Fasilitas stemming memanfaatkan algoritma Nazief & Adriani dari PySastrawi. Karena stemming membutuhkan beban komputasi morfologi yang cukup intensif, opsi ini disediakan secara fleksibel (*toggleable*) sehingga pengguna dapat memilih saat melakukan pengindeksan dokumen.

```python
def preprocess_text(text: str, use_stemming: bool = False) -> Tuple[str, List[str]]:
    """
    Pipeline lengkap preprocessing teks:
    1. Case Folding
    2. Tokenization
    3. Stopword Removal
    4. Stemming (opsional)
    """
    text_clean = case_folding(text)
    tokens = tokenize(text_clean)
    tokens_no_stop = remove_stopwords(tokens)

    if use_stemming and tokens_no_stop:
        final_tokens = stem_tokens(tokens_no_stop)
    else:
        final_tokens = tokens_no_stop

    clean_text = " ".join(final_tokens)
    return clean_text, final_tokens
```

---

### 4.2 Modul Pengindeksan Dokumen (`src/indexer.py`)
Modul ini bertugas membaca dataset berkas CSV, mengekstraksi kolom `judul` dan `konten`, melakukan preprocessing, dan menuliskan struktur posting list biner ke Whoosh.

```python
def build_index(
    sources: Optional[List[str]] = None,
    limit_per_source: Optional[int] = None,
    use_stemming: bool = False,
    progress_callback: Optional[Callable] = None,
    index_dir: str = DEFAULT_INDEX_DIR,
    dataset_dir: str = DATASET_DIR
) -> int:
    os.makedirs(index_dir, exist_ok=True)
    schema = get_schema()
    ix = create_in(index_dir, schema)
    writer = ix.writer(limitmb=256, multisegment=True)

    total_indexed = 0
    # Iterasi setiap dataset (etd-usk, etd-ugm, kompas, tempo, mojok)
    for source_name in sources_to_process:
        df = pd.read_csv(csv_path, dtype=str)
        # Tangani nilai NaN pada kolom judul dan konten
        df['judul'] = df['judul'].fillna('').astype(str)
        df['konten'] = df['konten'].fillna('').astype(str)

        for idx, row in df.iterrows():
            judul = row['judul'].strip()
            konten = row['konten'].strip()
            if not judul and not konten:
                continue

            # Gabungkan judul dan konten untuk field pencarian
            combined_raw = f"{judul} {konten}"
            clean_text, _ = preprocess_text(combined_raw, use_stemming=use_stemming)

            writer.add_document(
                doc_id=f"{source_name}_{idx}",
                source=source_name,
                judul=judul,
                konten=konten,
                clean_text=clean_text
            )
            total_indexed += 1

    writer.commit()
    return total_indexed
```

**Aspek Desain Utama pada Indexer**:
* **Optimasi Penulisan RAM**: Menggunakan parameter `writer(limitmb=256, multisegment=True)` agar proses indexing dokumen besar tidak membebani memori kerja sistem.
* **Mekanisme *Reuse Index***: Fungsi `is_index_built()` mengecek keberadaan file indeks di folder `indexdir/`. Jika indeks sudah pernah dibuat, sistem tidak perlu mengulang proses parsing CSV, melainkan langsung dapat dipakai dalam hitungan milidetik.
* **Fleksibilitas Batasan Dokumen**: Tersedia mode demo (1.000 dokumen per sumber = 5.000 total) untuk pengujian cepat serta mode penuh (seluruh 49.484 dokumen).

---

### 4.3 Modul Penelusuran dan Pemeringkatan (`src/retrieval.py`)
Inti dari mesin Information Retrieval terletak pada kelas `InformationRetrievalEngine`. Alur kerjanya terbagi menjadi beberapa tahapan logis:

```python
class InformationRetrievalEngine:
    def __init__(self, index_dir: str = DEFAULT_INDEX_DIR):
        self.index_dir = index_dir
        self._index = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_limit: int = 200,
        source_filter: Optional[str] = None,
        use_stemming: bool = False
    ) -> Tuple[List[SearchResult], Dict]:
        ix = self._get_index()

        # 1. Preprocessing Query pengguna
        clean_query, query_tokens = preprocess_text(query, use_stemming=use_stemming)
        if not clean_query.strip():
            return [], {'execution_status': 'empty_query'}

        # 2. Whoosh Candidate Retrieval
        candidate_docs = []
        candidate_clean_texts = []

        with ix.searcher() as searcher:
            # Gunakan OrGroup agar dokumen yang memuat minimal 1 kata query tetap terjaring
            parser = MultifieldParser(["clean_text", "judul"], schema=ix.schema, group=OrGroup.factory(0.9))
            parsed_q = parser.parse(clean_query)

            filter_query = Term("source", source_filter) if source_filter else None
            results = searcher.search(parsed_q, limit=candidate_limit, filter=filter_query)

            for hit in results:
                candidate_docs.append({
                    'doc_id': hit['doc_id'],
                    'source': hit['source'],
                    'judul': hit['judul'],
                    'konten': hit['konten'],
                    'clean_text': hit['clean_text']
                })
                candidate_clean_texts.append(hit['clean_text'])

        # 3. Representasi Ruang Vektor: Bag of Words (BoW)
        vectorizer = CountVectorizer()
        # Gabungkan query sebagai baris ke-0, disusul kandidat dokumen
        corpus = [clean_query] + candidate_clean_texts
        dtm = vectorizer.fit_transform(corpus)

        query_vector = dtm[0]          # Baris 0: Vektor Query
        document_vectors = dtm[1:]     # Baris 1..N: Vektor Dokumen Kandidat

        # 4. Perhitungan Cosine Similarity
        cos_scores = cosine_similarity(query_vector, document_vectors).flatten()

        # 5. Sorting Descending untuk Top-K
        sorted_indices = cos_scores.argsort()[::-1]

        final_results = []
        for rank, doc_idx in enumerate(sorted_indices[:top_k], start=1):
            score = float(cos_scores[doc_idx])
            doc = candidate_docs[doc_idx]
            snippet = extract_snippet(doc['konten'], query_tokens)
            final_results.append(SearchResult(
                rank=rank, score=score, doc_id=doc['doc_id'],
                source=doc['source'], judul=doc['judul'],
                konten=doc['konten'], snippet=snippet
            ))

        return final_results, metadata
```

#### Perumusan Matematis Cosine Similarity
Dalam ruang vektor berdimensi $V$ (sejumlah kosakata unik dalam sub-korpus gabungan query dan dokumen kandidat):
* Vektor query: $\vec{q} = (w_{1,q}, w_{2,q}, \dots, w_{V,q})$
* Vektor dokumen: $\vec{d} = (w_{1,d}, w_{2,d}, \dots, w_{V,d})$

Nilai kemiripan sudut kosinus dihitung dengan rumus:

$$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|} = \frac{\sum_{i=1}^{V} w_{i,q} \times w_{i,d}}{\sqrt{\sum_{i=1}^{V} w_{i,q}^2} \times \sqrt{\sum_{i=1}^{V} w_{i,d}^2}}$$

Nilai skor berada pada selang $[0, 1]$. Nilai $1.0$ menunjukkan bahwa distribusi frekuensi kata pada dokumen sepenuhnya proporsional dengan kata pada query, sementara nilai $0.0$ menandakan ortogonalitas sempurna (tidak ada kata yang beririsan).

#### Ekstraksi Cuplikan Konteks (*Snippet*)
Untuk meningkatkan keterbacaan hasil, sistem mengimplementasikan algoritma pencarian posisi token query pertama pada teks asli dokumen, lalu memotong cuplikan sekitar 240 karakter di sekitar kemunculan kata tersebut:

```python
def extract_snippet(text: str, query_tokens: List[str], max_len: int = 240) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text

    lower_text = text.lower()
    first_pos = -1
    for token in query_tokens:
        pos = lower_text.find(token)
        if pos != -1 and (first_pos == -1 or pos < first_pos):
            first_pos = pos

    start = 0 if (first_pos == -1 or first_pos < 50) else max(0, first_pos - 40)
    end = min(len(text), start + max_len)
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix
```

---

### 4.4 Modul Antarmuka CLI (`src/ui.py` & `main.py`)
Tampilan terminal dirancang dengan prinsip estetika modern:
1. **Pewarnaan Kode ANSI**: Memberikan pembeda visual antara header menu, status kesuksesan, skor relevansi, dan peringatan kesalahan.
2. **Visualisasi Skor Relevansi**: Setiap peringkat dilengkapi dengan visualisasi persentase dan bar visual `[████████░░░░░░░░░░░░]` yang memudahkan mata pengguna menilai bobot kemiripan dokumen secara instan.
3. **Penyaringan Sumber Dinamis**: Memungkinkan pengguna mencari di seluruh 5 dataset sekaligus atau membatasi pada satu dataset tertentu (misal hanya `etd-usk` atau `tempo`).
4. **Pembaca Teks Penuh**: Pengguna dapat memasukkan angka peringkat `[1-5]` untuk membaca isi lengkap artikel/abstrak yang diinginkan.

---

## 5. Pengujian Query dan Analisis Hasil

Untuk memvalidasi keandalan dan akurasi sistem, dilakukan serangkaian pengujian empiris menggunakan berbagai query dengan karakteristik berbeda: bidang pendidikan daring, topik ekonomi makro, dan topik finansial pasar modal.

### 5.1 Rangkuman Hasil Pengujian Empiris

Berikut adalah tabel rekapitulasi hasil eksekusi pengujian query pada sistem:

| No | Query Pengguna | Query Bersih (*Clean Query*) | Token yang Digunakan | Kandidat Terjaring (*Whoosh*) | Ukuran Kosakata (*Vocab BoW*) | Dokumen Peringkat 1 | Skor Cosine Similarity |
|:--:|:---|:---|:---|:---:|:---:|:---|:---:|
| 1 | `pembelajaran daring matematika` | `pembelajaran daring matematika` | `['pembelajaran', 'daring', 'matematika']` | 8 | 280 | *EFEKTIVITAS PEMBELAJARAN DARING MATA PELAJARAN BAHASA INDONESIA...* | **0.1127** |
| 2 | `inflasi dan pertumbuhan ekonomi` | `inflasi pertumbuhan ekonomi` | `['inflasi', 'pertumbuhan', 'ekonomi']` | 13 | 599 | *PENGARUH INFLASI DAN SUKU BUNGA TERHADAP PERTUMBUHAN EKONOMI INDONESIA* | **0.4157** |
| 3 | `investor saham bursa efek` | `investor saham bursa efek` | `['investor', 'saham', 'bursa', 'efek']` | 2 | 83 | *PENGUJIAN LOSS AVERSION INVESTOR DI BURSA EFEK INDONESIA...* | **0.3329** |

---

### 5.2 Analisis Rinci Kasus Uji 1: Topik Pendidikan Daring
* **Query Masukan**: `pembelajaran daring matematika`
* **Hasil Preprocessing**: Token `['pembelajaran', 'daring', 'matematika']` (tidak ada stopword yang dibuang karena semua kata bernilai informatif).
* **Kandidat Terpilih**: 8 dokumen.
* **Ukuran Dimensi Kosakata (*V*)**: 280 kata unik.

#### Rincian Peringkat Teratas:
1. **Rank 1 (Skor: 0.1127)**:
   - **Judul**: *EFEKTIVITAS PEMBELAJARAN DARING MATA PELAJARAN BAHASA INDONESIA SISWA KELAS XI SMK NEGERI 4 MEULABOH*
   - **Sumber**: `etd-usk`
   - **Cuplikan**: `ABSTRAKRahmi, Arinir. 2021. Efektivitas Pembelajaran Daring Mata Pelajaran Bahasa Indonesia Siswa Kelas XI SMK Negeri 4...`
2. **Rank 2 (Skor: 0.0543)**:
   - **Judul**: *PEMAHAMAN KONSEP DAN MOTIVASI PESERTA DIDIK SMP MELALUI PENDEKATAN SAINTIFIK BERBANTUAN TEKNIK MIND MAPPING DALAM PEMBELAJARAN MATEMATIKA*
   - **Sumber**: `etd-usk`
3. **Rank 3 (Skor: 0.0200)**:
   - **Judul**: *HASIL BELAJAR SISWA YANG MENGALAMI LEARNING LOSS MELALUI REMEDIAL BERBASIS E-LEARNING PADA MATERI BENTUK ALJABAR*
   - **Sumber**: `etd-usk`

**Analisis Ilmiah**:
Dokumen Rank 1 memuat dua kata kunci sekaligus, yaitu `pembelajaran` dan `daring` yang muncul berulang kali di bagian judul maupun abstrak, sehingga menghasilkan nilai dot product $\vec{q} \cdot \vec{d}$ yang lebih tinggi dibanding Rank 2 yang hanya memuat kata `pembelajaran` dan `matematika` dengan frekuensi lebih sedikit. Rank 3 menempati urutan paling bawah karena hanya memiliki relasi konseptual tidak langsung (`pembelajaran remedial`).

---

### 5.3 Analisis Rinci Kasus Uji 2: Topik Makroekonomi
* **Query Masukan**: `inflasi dan pertumbuhan ekonomi`
* **Hasil Preprocessing**: 
  - Kata sambung `dan` berhasil dideteksi sebagai stopword dan **dihapus**.
  - Token bersih: `['inflasi', 'pertumbuhan', 'ekonomi']`.
* **Kandidat Terpilih**: 13 dokumen.
* **Ukuran Dimensi Kosakata (*V*)**: 599 kata unik.

#### Rincian Peringkat Teratas:
1. **Rank 1 (Skor: 0.4157)**:
   - **Judul**: *PENGARUH INFLASI DAN SUKU BUNGA TERHADAP PERTUMBUHAN EKONOMI INDONESIA*
   - **Sumber**: `etd-usk`
   - **Cuplikan**: `...ini bertujuan untuk mengetahui pengaruh inflasi dan suku bunga terhadap pertumbuhan ekonomi Indonesia...`
2. **Rank 2 (Skor: 0.2946)**:
   - **Judul**: *PENGARUH PERKEMBANGAN INFRASTRUKTUR EKONOMI DAN SOSIAL TERHADAP PERTUMBUHAN EKONOMI DI PROVINSI ACEH*
   - **Sumber**: `etd-usk`
   - **Cuplikan**: `...melihat pengaruh perkembangan infrastruktur ekonomi dan sosial terhadap pertumbuhan ekonomi di Provinsi Aceh...`
3. **Rank 3 (Skor: 0.1516)**:
   - **Judul**: *PENGARUH PENDAPATAN ASLI DAERAH, DANA ALOKASI UMUM DAN DANA ALOKASI KHUSUS TERHADAP KEMISKINAN MELALUI PERTUMBUHAN EKONOMI DI PROVINSI ACEH*
   - **Sumber**: `etd-usk`

**Analisis Ilmiah**:
* Efektivitas pembersihan stopword terlihat sangat nyata pada pengujian ini. Jika kata `dan` tidak dibuang, dokumen dengan banyak kata `dan` akan mendapatkan lonjakan skor semu (*false relevance boost*).
* Dokumen Rank 1 mencapai skor kemiripan yang sangat tinggi (**0.4157** atau ~41.6%) karena ketiga term `inflasi`, `pertumbuhan`, dan `ekonomi` muncul secara simultan dan berulang kali pada judul dan paragraf utama.
* Dokumen Rank 2 hanya memuat kata `pertumbuhan` dan `ekonomi` tanpa kata `inflasi`, sehingga skornya secara proporsional turun menjadi **0.2946**.

---

### 5.4 Analisis Rinci Kasus Uji 3: Topik Pasar Modal
* **Query Masukan**: `investor saham bursa efek`
* **Hasil Preprocessing**: Token `['investor', 'saham', 'bursa', 'efek']`.
* **Kandidat Terpilih**: 2 dokumen.
* **Ukuran Dimensi Kosakata (*V*)**: 83 kata unik.

#### Rincian Peringkat Teratas:
1. **Rank 1 (Skor: 0.3329)**:
   - **Judul**: *PENGUJIAN LOSS AVERSION INVESTOR DI BURSA EFEK INDONESIA (STUDI PADA PERUSAHAAN INDEKS KOMPAS100 YANG TERCATAT DI BURSA EFEK INDONESIA)*
   - **Sumber**: `etd-usk`
   - **Cuplikan**: `...perusahaan Indeks Kompas100 yang tercatat di Bursa Efek Indonesia. Pengujian salah satu bias perilaku investor ini dilakukan...`
2. **Rank 2 (Skor: 0.0489)**:
   - **Judul**: *PENGARUH MANAGERIAL OVERCONFIDENCE TERHADAP PENGUNGKAPAN CORPORATE SOCIAL RESPONSIBILITY DALAM UPAYA MENINGKATKAN NILAI PERUSAHAAN*
   - **Sumber**: `etd-usk`
   - **Cuplikan**: `...perusahaan nonkeuangan yang terdaftar di Bursa Efek Indonesia (BEI) pada tahun 2018 hingga 2022...`

**Analisis Ilmiah**:
* Terjadi pemisahan skor yang sangat tajam antara peringkat 1 (0.3329) dan peringkat 2 (0.0489). Dokumen Rank 1 memuat istilah majemuk `Bursa Efek Indonesia` dan kata `investor` secara intensif. Sebaliknya, Rank 2 hanya kebetulan menyebut frasa `Bursa Efek Indonesia` dalam satu kalimat latar belakang sampel tanpa memuat kata `investor` maupun `saham`.
* Hal ini membuktikan bahwa perkalian titik vektor pada *Cosine Similarity* secara efektif menghukum dokumen yang hanya memuat sebagian kecil dari kata pencarian.

---

### 5.5 Evaluasi Kinerja Waktu Komputasi (Latency)
Berdasarkan log pencarian pada mesin uji, waktu yang dihabiskan untuk satu siklus penelusuran lengkap (Preprocessing + Whoosh Retrieval + CountVectorizer + Cosine Similarity + Ranking + Snippet Extraction) berkisar antara **15 ms hingga 45 ms**. Kecepatan respons ini memenuhi kriteria standar *real-time search* pada aplikasi interaktif, yang membuktikan keberhasilan arsitektur *two-stage retrieval*.

---

## 6. Kesimpulan

### 6.1 Kesimpulan Proyek
Berdasarkan seluruh tahapan perancangan, implementasi, dan pengujian empiris yang telah dilakukan, dapat disimpulkan bahwa:
1. **Efektivitas Arsitektur Two-Stage**: Penggabungan *Whoosh Inverted Index* sebagai penyaring kandidat cepat dengan model *Bag of Words (CountVectorizer)* dan *Cosine Similarity* berhasil menyajikan hasil penelusuran yang akurat dengan waktu respons sub-detik (< 50 milidetik).
2. **Peran Kritis Text Preprocessing**: Penghapusan stopword terbukti sangat krusial dalam mencegah distorsi bobot pada perhitungan kemiripan vektor kosinus, memastikan bahwa kata kunci bermakna substantif yang menentukan posisi peringkat.
3. **Modularitas dan Usabilitas**: Pemisahan sistem menjadi modul-modul independen (`preprocessing.py`, `indexer.py`, `retrieval.py`, `ui.py`, `main.py`) menghasilkan basis kode yang bersih, mudah dipelihara, dan memiliki antarmuka interaktif yang nyaman digunakan untuk menjelajahi puluhan ribu dokumen dari 5 sumber dataset.

### 6.2 Saran dan Pengembangan Lanjutan
Untuk pengembangan sistem Information Retrieval ini di masa mendatang, beberapa saran peningkatan yang dapat diterapkan antara lain:
1. **Penerapan Pembobotan TF-IDF / BM25 Langsung pada Matriks Vektor**: Menambahkan komponen *Inverse Document Frequency* (IDF) pada ruang vektor agar kata-kata yang sangat langka di seluruh koleksi mendapatkan bobot relevansi yang lebih tinggi secara proporsional.
2. **Dense Vector Embeddings (Semantic Search)**: Mengintegrasikan model representasi berbasis transformer berbahasa Indonesia (seperti IndoBERT atau Sentence-BERT) untuk menangani permasalahan sinonimi (*vocabulary mismatch*), sehingga kata seperti `ponsel` dan `telepon pintar` dapat dikenali kemiripan maknanya.
3. **Ekspansi Query Otomatis (Query Expansion)**: Menyediakan mekanisme saran kata kunci pelengkap atau peluasan query menggunakan tesaurus bahasa Indonesia guna meningkatkan nilai perolehan kembali (*recall*).
