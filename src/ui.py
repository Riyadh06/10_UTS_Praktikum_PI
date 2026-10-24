"""
Module antarmuka CLI, pewarnaan terminal, dan visualisasi hasil pencarian.
Menggunakan colorama untuk pewarnaan cross-platform (Windows & Linux).
"""

import os
import sys
import re
from typing import List, Dict
from colorama import init, Fore, Back, Style

# Inisialisasi colorama
init(autoreset=True)

# Memastikan UTF-8 jika didukung
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CYAN = Fore.CYAN
BRIGHT_CYAN = Fore.CYAN + Style.BRIGHT
GREEN = Fore.GREEN
BRIGHT_GREEN = Fore.GREEN + Style.BRIGHT
YELLOW = Fore.YELLOW
BRIGHT_YELLOW = Fore.YELLOW + Style.BRIGHT
RED = Fore.RED
BRIGHT_RED = Fore.RED + Style.BRIGHT
MAGENTA = Fore.MAGENTA
BRIGHT_MAGENTA = Fore.MAGENTA + Style.BRIGHT
BLUE = Fore.BLUE
WHITE = Fore.WHITE
DIM = Style.DIM
RESET = Style.RESET_ALL
BOLD = Style.BRIGHT


def clear_screen():
    """Membersihkan layar terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Menampilkan banner header sistem IR."""
    banner = f"""
{BRIGHT_CYAN}+======================================================================+
|                   {BRIGHT_YELLOW}INFORMATION RETRIEVAL (IR) SYSTEM{BRIGHT_CYAN}                  |
|      {WHITE}Whoosh Indexing *Bag of Words (BoW)*  Cosine Similarity{BRIGHT_CYAN}        |
+======================================================================+{RESET}
"""
    print(banner)


def print_main_menu():
    """
    Menampilkan menu utama sesuai instruksi teknis:
    === INFORMATION RETRIEVAL SYSTEM ===
    [1] Load & Index Dataset
    [2] Search Query
    [3] Exit
    ====================================
    """
    menu = f"""{BRIGHT_CYAN}=== INFORMATION RETRIEVAL SYSTEM ==={RESET}
{BRIGHT_GREEN}[1]{RESET} Load & Index Dataset
{BRIGHT_GREEN}[2]{RESET} Search Query
{BRIGHT_GREEN}[3]{RESET} Exit
{BRIGHT_CYAN}===================================={RESET}"""
    print(menu)


def print_source_badge(source: str) -> str:
    """Format badge warna untuk sumber dataset."""
    badges = {
        'etd-usk': f"{Fore.BLACK}{Back.GREEN} ETD-USK {RESET}",
        'etd-ugm': f"{Fore.BLACK}{Back.BLUE} ETD-UGM {RESET}",
        'kompas': f"{Fore.BLACK}{Back.CYAN} KOMPAS {RESET}",
        'tempo': f"{Fore.BLACK}{Back.MAGENTA} TEMPO {RESET}",
        'mojok': f"{Fore.BLACK}{Back.YELLOW} MOJOK {RESET}",
    }
    return badges.get(source, f"{Fore.WHITE}[{source.upper()}]{RESET}")


def highlight_terms(text: str, query_tokens: List[str]) -> str:
    """Menyorot (highlight) kemunculan kata kunci query dalam teks."""
    if not text or not query_tokens:
        return text

    # Buat regex pattern untuk seluruh query tokens (case-insensitive)
    escaped_tokens = [re.escape(tok) for tok in query_tokens if len(tok) >= 2]
    if not escaped_tokens:
        return text

    pattern = re.compile(r'\b(' + '|'.join(escaped_tokens) + r')\b', re.IGNORECASE)

    def replace_func(match):
        return f"{BRIGHT_YELLOW}{match.group(0)}{RESET}"

    return pattern.sub(replace_func, text)


def safe_str(text: str) -> str:
    """Membersihkan karakter yang tidak dapat di-encode pada encoding terminal saat ini."""
    if not isinstance(text, str):
        text = str(text or "")
    encoding = sys.stdout.encoding or 'utf-8'
    try:
        return text.encode(encoding, errors='replace').decode(encoding)
    except Exception:
        return text.encode('ascii', errors='replace').decode('ascii')


def display_results(results: List, metadata: Dict):
    """Menampilkan hasil ranking pencarian dokumen dalam format terstruktur."""
    print(f"\n{BRIGHT_CYAN}--- DETAIL PROSES PENELUSURAN INFORMASI (IR PIPELINE) ---{RESET}")
    print(f"* {BOLD}Query Asli:{RESET} {safe_str(metadata.get('original_query', ''))}")
    print(f"* {BOLD}Query Bersih (Preprocessed):{RESET} {GREEN}{safe_str(metadata.get('clean_query', ''))}{RESET}")
    print(f"* {BOLD}Token Kata Kunci:{RESET} {metadata.get('query_tokens')}")
    print(f"* {BOLD}Kandidat Whoosh Ditemukan:{RESET} {metadata.get('num_candidates_retrieved')} dokumen")
    print(f"* {BOLD}Dimensi Kosakata (BoW Vocab):{RESET} {metadata.get('vocabulary_size')} fitur/term unik")
    print(f"{BRIGHT_CYAN}----------------------------------------------------------{RESET}\n")

    if not results:
        print(f"{BRIGHT_RED}[!] Tidak ada dokumen yang relevan dengan query tersebut.{RESET}")
        print(f"{DIM}Saran: Coba gunakan kata kunci umum atau sinonim lain.{RESET}\n")
        return

    print(f"{BRIGHT_GREEN}=== TOP 5 DOKUMEN DENGAN KOSINE SIMILARITY TERTINGGI ==={RESET}\n")
    query_tokens = metadata.get('query_tokens', [])

    for doc in results:
        badge = print_source_badge(doc.source)
        score_percent = doc.score * 100
        score_bar = "#" * int(doc.score * 15) + "-" * (15 - int(doc.score * 15))

        clean_title = safe_str(doc.judul)
        clean_snippet = safe_str(doc.snippet)

        print(f"{BRIGHT_YELLOW}Peringkat #{doc.rank}{RESET} {badge} {BRIGHT_CYAN}ID: {doc.doc_id}{RESET}")
        print(f"{BOLD}Judul :{RESET} {highlight_terms(clean_title, query_tokens)}")
        print(f"{BOLD}Skor  :{RESET} {BRIGHT_GREEN}{doc.score:.4f}{RESET} ({score_percent:.1f}%) [{BRIGHT_GREEN}{score_bar}{RESET}]")
        print(f"{BOLD}Kutipan (Snippet):{RESET}")
        print(f"  {DIM}\"{RESET}{highlight_terms(clean_snippet, query_tokens)}{DIM}\"{RESET}")
        print(f"{DIM}{'-' * 70}{RESET}")


def view_document_details(doc):
    """Menampilkan isi lengkap dokumen terpilih."""
    print(f"\n{BRIGHT_CYAN}+======================================================================+{RESET}")
    print(f"| {BRIGHT_YELLOW}DOKUMEN DETAIL [Peringkat #{doc.rank}] - {doc.doc_id}{RESET}")
    print(f"{BRIGHT_CYAN}+======================================================================+{RESET}")
    print(f"{BOLD}Sumber  :{RESET} {doc.source.upper()}")
    print(f"{BOLD}Skor    :{RESET} {doc.score:.4f}")
    print(f"{BOLD}Judul   :{RESET} {safe_str(doc.judul)}")
    print(f"\n{BOLD}Konten Lengkap:{RESET}")
    print(f"{DIM}{'─' * 70}{RESET}")
    print(safe_str(doc.konten))
    print(f"{DIM}{'─' * 70}{RESET}\n")
