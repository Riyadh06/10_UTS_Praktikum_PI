# Sistem Penelusuran Informasi (Information Retrieval System - CLI)

Sistem Information Retrieval (IR) berbasis Command-Line Interface (CLI) interaktif yang dirancang untuk melakukan proses indexing, penelusuran (retrieval), dan pemeringkatan (ranking) dokumen dari 5 sumber dataset nyata berbahasa Indonesia menggunakan pendekatan **Vector Space Model (Bag of Words)**, **Whoosh Inverted Index**, dan **Cosine Similarity**.

---

## 📚 5 Dataset Utama

Sistem mengintegrasikan dan mendukung pencarian lintas 5 sumber teks nyata:

| No | Dataset | Sumber Teks | Format Kolom | Jumlah Dokumen |
|:--:|:--------|:------------|:-------------|:--------------:|
| 1 | `etd-usk` | Tesis & Disertasi Universitas Syiah Kuala | `judul`, `konten` | 10.000 |
| 2 | `etd-ugm` | Tesis & Disertasi Universitas Gadjah Mada | `judul`, `konten` | 10.000 |
| 3 | `kompas` | Berita Harian Nasional Kompas | `judul`, `konten` | 10.000 |
| 4 | `tempo` | Majalah Berita & Opini Tempo | `judul`, `konten` | 10.000 |
| 5 | `mojok` | Esai, Artikel Populer & Satir Mojok.co | `judul`, `konten` | 9.484 |
| **Total** | | | | **49.484 Dokumen** |

---

## 🛠️ Pipeline Information Retrieval

Sistem menerapkan arsitektur dua tahap (*Two-Stage IR Retrieval & Reranking*):

```text
[Input Query Pengguna]
         │
         ▼
[1. Text Preprocessing]
  ├── Case Folding (lowercase normalisasi)
  ├── Tokenization (ekstraksi kata alfabetik)
  ├── Stopword Removal (PySastrawi 809 kata + istilah umum)
  └── Stemming (Opsional: PySastrawi Nazief & Adriani)
         │
         ▼
[2. Whoosh Inverted Index Search]
  └── Mengambil kandidat dokumen yang mengandung term query (OrGroup coordination)
         │
         ▼
[3. Representasi Vektor: Bag of Words (BoW)]
  └── Ekstraksi Term-Document Matrix via scikit-learn CountVectorizer
         │
         ▼
[4. Perhitungan Kemiripan: Cosine Similarity]
  └── cos_sim(q, d) = (q · d) / (||q|| * ||d||)
         │
         ▼
[5. Ranking & Display Top-5 Dokumen]
  ├── Urutkan berdasarkan skor cosine similarity tertinggi (Descending)
  ├── Tampilkan Rank, Skor, Sumber Dataset, Judul, dan Snippet
  └── Fitur interaktif membaca dokumen penuh
```

---

## 📂 Struktur Direktori

```text
UTS/
├── dataset/                    # Direktori 5 file CSV dataset
│   ├── etd_ugm.csv
│   ├── etd_usk.csv
│   ├── kompas.csv
│   ├── mojok.csv
│   └── tempo.csv
├── indexdir/                   # Direktori penyimpanan indeks Whoosh
├── src/                        # Modular Source Code
│   ├── __init__.py
│   ├── preprocessing.py        # Case folding, tokenisasi, stopword, stemming
│   ├── indexer.py              # Skema Whoosh, pembacaan CSV, batch indexing
│   ├── retrieval.py            # Mesin penelusuran, BoW, & Cosine Similarity
│   └── ui.py                   # Antarmuka CLI, pewarnaan terminal, visualisasi
├── main.py                     # Entry point program CLI interaktif
├── requirements.txt            # Dependensi pustaka Python
└── README.md                   # Dokumentasi lengkap proyek
```

---

## 🚀 Panduan Instalasi & Menjalankan Program

### 1. Prasyarat Lingkungan
Pastikan Python 3.9+ sudah terpasang di sistem Anda.

### 2. Instalasi Dependensi
Jalankan perintah berikut di terminal:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan CLI Interaktif
Jalankan program utama:
```bash
python main.py
```

---

## 🖥️ Panduan Penggunaan Antarmuka CLI

Saat program dijalankan, menu utama akan tampil:

```text
╔══════════════════════════════════════════════════════════════════════╗
║                   INFORMATION RETRIEVAL (IR) SYSTEM                  ║
║      Whoosh Indexing • Bag of Words (BoW) • Cosine Similarity        ║
╚══════════════════════════════════════════════════════════════════════╝

=== INFORMATION RETRIEVAL SYSTEM ===
[1] Load & Index Dataset
[2] Search Query
[3] Exit
====================================

Pilih Menu [1/2/3]:
```

### Menu [1]: Load & Index Dataset
Pilihan pembuatan indeks dokumen:
- **Mode Demo / Cepat (Rekomendasi Uji Coba):** Mengindeks 1.000 dokumen per dataset (total 5.000 dokumen), selesai dalam ~20-30 detik.
- **Mode Lengkap:** Mengindeks seluruh 49.484 dokumen dari kelima dataset (~2-3 menit).
- **Mode Custom:** Memilih dataset tertentu (misal hanya `kompas` dan `tempo`).
- **Reuse Index:** Jika indeks sudah ada di disk (`indexdir/`), sistem secara otomatis mendeteksi dan dapat langsung digunakan tanpa perlu indeks ulang.

### Menu [2]: Search Query
1. Masukkan kata kunci pencarian (contoh: `kecerdasan buatan teknologi`, `pemilu presiden politik`, `pendidikan karakter anak`).
2. Pilih filter sumber (Semua Dataset, atau pilih salah satu dari 5 dataset).
3. Sistem akan memproses query melalui pipeline IR dan menampilkan:
   - Metadata pemrosesan teks (Query bersih, token, jumlah kandidat Whoosh, ukuran kosakata BoW).
   - **Top 5 Dokumen Teratas** dengan skor Cosine Similarity tertinggi, lengkap dengan persentase kemiripan, bar grafis, dan cuplikan teks (*snippet*) yang menyorot kata kunci.
4. Pengguna dapat memilih nomor peringkat `[1-5]` untuk membaca isi artikel/dokumen secara lengkap.

### Menu [3]: Exit
Keluar dari program dengan aman.

---

## 📊 Formulasi Teoretis

### 1. Bag of Words (BoW) via CountVectorizer
Dokumen dan query direpresentasikan sebagai vektor frekuensi kemunculan term dalam ruang berdimensi $V$ (jumlah kosakata unik):
$$\vec{d} = (tf_{1,d}, tf_{2,d}, \dots, tf_{V,d})$$
$$\vec{q} = (tf_{1,q}, tf_{2,q}, \dots, tf_{V,q})$$

### 2. Cosine Similarity
Menghitung sudut kosinus antara vektor query $\vec{q}$ dan vektor dokumen $\vec{d}$:
$$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|} = \frac{\sum_{i=1}^{V} q_i \cdot d_i}{\sqrt{\sum_{i=1}^{V} q_i^2} \cdot \sqrt{\sum_{i=1}^{V} d_i^2}}$$
Nilai kemiripan berada pada rentang $[0, 1]$, di mana nilai mendekati 1 merepresentasikan tingkat relevansi topik yang sangat tinggi.
