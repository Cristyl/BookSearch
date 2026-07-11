# quality_test.py — valuta la qualità della ricerca con il test di self-retrieval:
# un estratto del testo di un libro indicizzato, usato come query, deve restituire
# quel libro ai primi posti.
#
# Per ogni size campiona N libri dal CSV pulito, estrae brani di lunghezza variabile
# dal CENTRO del testo (per evitare l'header standard di Gutenberg, uguale per tutti)
# e misura a che rank il motore restituisce il libro atteso.
#
# Metriche: Top-1 (rank 1), Top-5 (rank <= 5), MRR (media di 1/rank, 0 se fuori dai top 10).
#
# Uso:
#   python quality_test.py                          # tutte le size, 10 libri, 50/200/1000 parole
#   python quality_test.py --sizes 5000 --sample 30
#   python quality_test.py --lengths 100 500
#
# Output: tabella a video + quality_results.csv con le misure grezze.
# Nota: legge i CSV in data/ (quello da 5000 pesa ~1.5 GB: qualche minuto di scansione).

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ALL_SIZES = ["500", "2000", "5000"]


def api(base_url, path, params=None, timeout=600):
    url = base_url + path + (("?" + urllib.parse.urlencode(params)) if params else "")
    with urllib.request.urlopen(url, timeout=timeout) as res:
        return json.load(res)


def sample_books(csv_path, n, seed):
    """Campionamento uniforme (reservoir sampling) leggendo il CSV a chunk."""
    rng = random.Random(seed)
    sample, seen = [], 0
    for chunk in pd.read_csv(csv_path, usecols=["Title", "Text"], chunksize=500):
        chunk = chunk.dropna(subset=["Title", "Text"])
        for _, row in chunk.iterrows():
            seen += 1
            item = (str(row["Title"]).strip(), str(row["Text"]))
            if len(sample) < n:
                sample.append(item)
            else:
                j = rng.randrange(seen)
                if j < n:
                    sample[j] = item
    return sample


def excerpt(text, n_words):
    """Brano di n_words parole preso dal centro del testo."""
    words = text.split()
    if len(words) <= n_words:
        return " ".join(words)
    start = max(0, len(words) // 2 - n_words // 2)
    return " ".join(words[start:start + n_words])


def main():
    parser = argparse.ArgumentParser(description="Test di qualità self-retrieval.")
    parser.add_argument("--base-url", default="http://localhost:5000")
    parser.add_argument("--sizes", nargs="*", default=ALL_SIZES, choices=ALL_SIZES)
    parser.add_argument("--sample", type=int, default=10, help="libri campionati per size (default 10)")
    parser.add_argument("--lengths", nargs="*", type=int, default=[50, 200, 1000],
                        help="lunghezze dei brani in parole (default 50 200 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="quality_results.csv")
    args = parser.parse_args()

    try:
        api(args.base_url, "/api/health", timeout=10)
    except Exception as e:
        print(f"ERRORE: backend non raggiungibile su {args.base_url} ({e})")
        sys.exit(1)

    raw_rows = []  # (size, title, n_words, rank)

    for size in sorted(args.sizes, key=int):
        csv_path = os.path.join(BASE_DIR, "data", f"gutenberg_data_{size}_clean.csv")
        if not os.path.isfile(csv_path):
            print(f"Dataset {size}: {csv_path} non trovato, salto.")
            continue

        print(f"\n{'='*60}\nDataset {size}: campiono {args.sample} libri dal CSV...\n{'='*60}")
        t0 = time.time()
        books = sample_books(csv_path, args.sample, args.seed)
        print(f"Campionati {len(books)} libri in {time.time()-t0:.0f}s")

        for b_idx, (title, text) in enumerate(books, 1):
            for n_words in args.lengths:
                q = excerpt(text, n_words)
                try:
                    data = api(args.base_url, "/api/search",
                               {"q": q, "top": 10, "dataset": size})
                except Exception as e:
                    print(f"  ERRORE query per '{title[:40]}' ({n_words}w): {e}")
                    raw_rows.append((size, title, n_words, None))
                    continue
                rank = next((r["rank"] for r in data["results"]
                             if (r["title"] or "").strip() == title), None)
                raw_rows.append((size, title, n_words, rank))
            print(f"  libro {b_idx}/{len(books)}: {title[:50]}")

    # ── Metriche ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nRISULTATI\n{'='*60}")
    print("| Catalogo | Parole query | Top-1 | Top-5 | MRR | N |")
    print("|---|---|---|---|---|---|")
    for size in sorted(args.sizes, key=int):
        for n_words in args.lengths:
            ranks = [r for (s, _, w, r) in raw_rows if s == size and w == n_words]
            if not ranks and not any(s == size for (s, _, _, _) in raw_rows):
                continue
            n     = len(ranks)
            if n == 0:
                continue
            top1  = sum(1 for r in ranks if r == 1) / n
            top5  = sum(1 for r in ranks if r is not None and r <= 5) / n
            mrr   = sum((1 / r) if r else 0.0 for r in ranks) / n
            print(f"| {size} | {n_words} | {top1:.0%} | {top5:.0%} | {mrr:.3f} | {n} |")

    misses = [(s, t, w) for (s, t, w, r) in raw_rows if r is None]
    if misses:
        print("\nFuori dai top 10:")
        for s, t, w in misses:
            print(f"  - [{s}, {w} parole] {t}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "title", "query_words", "rank"])
        w.writerows(raw_rows)
    print(f"\nMisure grezze salvate in {args.out}")


if __name__ == "__main__":
    main()
