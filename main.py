import sys
import os
import time

# Memastikan output Windows terminal mendukung UTF-8
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.preprocessing import preprocess_text
from src.indexer import (
    DATASET_FILES,
    DEFAULT_INDEX_DIR,
    is_index_built,
    get_index_stats,
    get_dataset_inventory,
    build_index
)
from src.retrieval import InformationRetrievalEngine
from src.ui import (
    print_banner,
    print_main_menu,
    display_results,
    view_document_details,
    CYAN,
    BRIGHT_CYAN,
    GREEN,
    BRIGHT_GREEN,
    YELLOW,
    BRIGHT_YELLOW,
    RED,
    BRIGHT_RED,
    WHITE,
    DIM,
    RESET,
    BOLD
)


def handle_index_dataset():
    """Menangani alur [1] Load & Index Dataset."""
    print(f"\n{BRIGHT_CYAN}--- [1] LOAD & INDEX DATASET ---{RESET}")
    
    # Cek status inventaris file dataset CSV
    inventory = get_dataset_inventory()
    print(f"\n{BOLD}Daftar 5 Dataset Utama:{RESET}")
    for idx, (key, info) in enumerate(inventory.items(), start=1):
        status_str = f"{GREEN}Tersedia ({info['size_mb']} MB){RESET}" if info['available'] else f"{RED}Tidak Ditemukan{RESET}"
        print(f"  {idx}. {BOLD}{key:<8}{RESET} : {info['filename']:<15} -> {status_str}")

    # Cek apakah index sudah pernah dibuat
    already_indexed = is_index_built()
    if already_indexed:
        stats = get_index_stats()
        print(f"\n{BRIGHT_GREEN}[OK] Indeks Whoosh ditemukan pada direktori 'indexdir/'!{RESET}")
        print(f"    Total Dokumen Terindeks: {BRIGHT_YELLOW}{stats['total_docs']:,}{RESET} dokumen.")
        print(f"\nPilihan Aksi:")
        print(f"  {BRIGHT_GREEN}[1]{RESET} Gunakan indeks yang sudah ada (Kembali ke menu)")
        print(f"  {BRIGHT_GREEN}[2]{RESET} Buat ulang indeks LENGKAP (Semua 5 Dataset: ~49.484 dokumen)")
        print(f"  {BRIGHT_GREEN}[3]{RESET} Buat indeks MODE CEPAT / DEMO (1.000 dokumen per dataset = 5.000 dokumen)")
        print(f"  {BRIGHT_GREEN}[4]{RESET} Buat indeks dataset TERTENTU")
        
        choice = input(f"\n{BOLD}Pilih opsi [1/2/3/4] (default=1): {RESET}").strip()
        if choice in ('1', '', 'b', 'kembali'):
            print(f"{GREEN}[OK] Menggunakan indeks yang sudah tersimpan.{RESET}\n")
            return
        elif choice == '2':
            sources = list(DATASET_FILES.keys())
            limit = None
        elif choice == '3':
            sources = list(DATASET_FILES.keys())
            limit = 1000
        elif choice == '4':
            sources, limit = prompt_custom_dataset()
            if not sources:
                return
        else:
            print(f"{RED}[!] Pilihan tidak valid. Kembali ke menu utama.{RESET}\n")
            return
    else:
        print(f"\n{YELLOW}[!] Belum ada indeks dokumen. Silakan pilih mode pembuatan indeks:{RESET}")
        print(f"  {BRIGHT_GREEN}[1]{RESET} Indeks MODE CEPAT / DEMO (1.000 dokumen per dataset = 5.000 dokumen) [Disarankan untuk tes cepat]")
        print(f"  {BRIGHT_GREEN}[2]{RESET} Indeks LENGKAP (Semua 5 Dataset: ~49.484 dokumen) [Butuh waktu ~2-3 menit]")
        print(f"  {BRIGHT_GREEN}[3]{RESET} Indeks Dataset TERTENTU")
        print(f"  {BRIGHT_GREEN}[4]{RESET} Kembali ke Menu Utama")

        choice = input(f"\n{BOLD}Pilih opsi [1/2/3/4] (default=1): {RESET}").strip()
        if choice in ('1', ''):
            sources = list(DATASET_FILES.keys())
            limit = 1000
        elif choice == '2':
            sources = list(DATASET_FILES.keys())
            limit = None
        elif choice == '3':
            sources, limit = prompt_custom_dataset()
            if not sources:
                return
        elif choice == '4':
            return
        else:
            print(f"{RED}[!] Pilihan tidak valid.{RESET}\n")
            return

    # Opsi Stemming
    print(f"\n{BOLD}Opsi Stemming (PySastrawi):{RESET}")
    print(f"{DIM}Catatan: Stemming meningkatkan recall kata dasar, namun menambah waktu proses indexing.{RESET}")
    stem_choice = input(f"Aktifkan stemming saat indexing? (y/N, default=N): ").strip().lower()
    use_stemming = stem_choice == 'y'

    # Eksekusi pembuatan indeks
    print(f"\n{BRIGHT_CYAN}[*] Memulai proses preprocessing dan indexing Whoosh...{RESET}")
    start_time = time.time()

    def progress(source_name, current, total, elapsed):
        speed = current / elapsed if elapsed > 0 else 0
        pct = (current / total) * 100
        print(f"  -> [{source_name}] {current:,}/{total:,} ({pct:5.1f}%) | {elapsed:.1f}s ({speed:.0f} dok/s)", end='\r')

    try:
        total = build_index(
            sources=sources,
            limit_per_source=limit,
            use_stemming=use_stemming,
            progress_callback=progress
        )
        total_time = time.time() - start_time
        print(f"\n\n{BRIGHT_GREEN}[OK] SUKSES! Sebanyak {total:,} dokumen berhasil di-preprocess dan diindeks.{RESET}")
        print(f"    Waktu eksekusi: {total_time:.2f} detik.")
        print(f"    Lokasi indeks : {DEFAULT_INDEX_DIR}\n")
    except Exception as e:
        print(f"\n{BRIGHT_RED}[!] Terjadi kesalahan saat indexing: {e}{RESET}\n")


def prompt_custom_dataset():
    """Dialog interaktif untuk memilih dataset spesifik."""
    print(f"\nPilih dataset yang ingin diindeks:")
    keys = list(DATASET_FILES.keys())
    for idx, key in enumerate(keys, 1):
        print(f"  [{idx}] {key}")
    print(f"  [0] Batal")

    sel = input(f"{BOLD}Masukkan nomor dataset (pisahkan koma jika lebih dari satu, contoh: 1,3): {RESET}").strip()
    if sel in ('0', '', 'batal'):
        return [], None

    selected_sources = []
    for part in sel.split(','):
        part = part.strip()
        if part.isdigit():
            i = int(part) - 1
            if 0 <= i < len(keys):
                selected_sources.append(keys[i])

    if not selected_sources:
        print(f"{RED}[!] Tidak ada dataset yang dipilih.{RESET}")
        return [], None

    limit_input = input(f"{BOLD}Batas jumlah dokumen per dataset (tekan Enter untuk semua): {RESET}").strip()
    limit = int(limit_input) if limit_input.isdigit() else None

    return selected_sources, limit


def handle_search_query(engine: InformationRetrievalEngine):
    """Menangani alur [2] Search Query."""
    print(f"\n{BRIGHT_CYAN}--- [2] SEARCH QUERY ---{RESET}")

    if not is_index_built():
        print(f"{BRIGHT_RED}[!] Indeks Whoosh belum dibuat!{RESET}")
        print(f"{YELLOW}Silakan jalankan menu [1] Load & Index Dataset terlebih dahulu.{RESET}\n")
        return

    # Loop pencarian interaktif
    while True:
        query_input = input(f"\n{BRIGHT_GREEN}Masukkan Query Pencarian{RESET} (atau ketik {YELLOW}'menu'{RESET} untuk kembali): ").strip()
        
        if not query_input:
            continue
        if query_input.lower() in ('menu', 'kembali', 'exit', 'q'):
            break

        # Filter dataset opsional
        print(f"{DIM}Cari di: [0] Semua Dataset | [1] etd-usk | [2] etd-ugm | [3] kompas | [4] tempo | [5] mojok{RESET}")
        source_opt = input(f"{DIM}Pilihan filter sumber (default=0): {RESET}").strip()
        source_map = {'1': 'etd-usk', '2': 'etd-ugm', '3': 'kompas', '4': 'tempo', '5': 'mojok'}
        source_filter = source_map.get(source_opt, None)

        print(f"\n{CYAN}[*] Memproses query dan menghitung kemiripan Cosine Similarity...{RESET}")
        t0 = time.time()
        results, metadata = engine.search(
            query=query_input,
            top_k=5,
            candidate_limit=250,
            source_filter=source_filter
        )
        t_elapsed = time.time() - t0

        display_results(results, metadata)
        print(f"{DIM}Waktu pencarian: {t_elapsed * 1000:.2f} ms{RESET}\n")

        # Sub-opsi melihat detail dokumen
        if results:
            while True:
                detail_opt = input(f"{BOLD}Pilih nomor peringkat [1-{len(results)}] untuk membaca dokumen penuh, atau [0] untuk query baru: {RESET}").strip()
                if detail_opt == '0' or detail_opt == '':
                    break
                if detail_opt.isdigit() and 1 <= int(detail_opt) <= len(results):
                    selected_doc = results[int(detail_opt) - 1]
                    view_document_details(selected_doc)
                else:
                    print(f"{YELLOW}[!] Masukkan angka antara 1 dan {len(results)}, atau 0.{RESET}")


def main():
    """Fungsi utama loop CLI."""
    engine = InformationRetrievalEngine()

    while True:
        print_banner()
        print_main_menu()
        
        try:
            pilihan = input(f"\n{BOLD}Pilih Menu [1/2/3]: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{YELLOW}Program dihentikan oleh pengguna. Keluar...{RESET}")
            sys.exit(0)

        if pilihan == '1':
            handle_index_dataset()
        elif pilihan == '2':
            handle_search_query(engine)
        elif pilihan == '3':
            print(f"\n{BRIGHT_GREEN}===================================================={RESET}")
            print(f"{BRIGHT_CYAN}Terima kasih telah menggunakan Information Retrieval System!{RESET}")
            print(f"{DIM}Sistem selesai. Sampai jumpa!{RESET}")
            print(f"{BRIGHT_GREEN}===================================================={RESET}\n")
            sys.exit(0)
        else:
            print(f"{BRIGHT_RED}[!] Pilihan tidak valid. Silakan masukkan 1, 2, atau 3.{RESET}\n")
            time.sleep(1)


if __name__ == '__main__':
    main()
