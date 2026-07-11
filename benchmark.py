# benchmark.py — misura la scalabilità della ricerca sui tre cataloghi.
#
# Per ogni size misura:
#   - tempo della PRIMA query (a cache fredda include il caricamento lazy del dataset)
#   - tempo medio +/- deviazione standard a cache calda, su una batteria di query ripetute
#
# Per misurare davvero la cache fredda serve riavviare il backend prima: --restart
# lo fa automaticamente (richiede docker nel PATH).
#
# Uso:
#   python benchmark.py                       # solo tempi a caldo (se già caricato)
#   python benchmark.py --restart             # riavvia il backend e misura anche il freddo
#   python benchmark.py --reps 5 --sizes 500 2000
#
# Output: tabella a video + benchmark_results.csv con le misure grezze.

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request

QUERIES = [
    "a scientist creates a living creature from dead body parts",
    "a vampire count in Transylvania drinking blood",
    "pirates searching for buried treasure with a map",
    "a detective solving mysteries in London",
    "a man shipwrecked alone on a desert island",
    "a journey to the center of the earth",
    "martians invade the earth with war machines",
    "a girl falls down a rabbit hole into a strange world",
    "a boy rafting down the Mississippi river",
    "an orphan boy among pickpockets in London",
    "a man travels through time to the distant future",
    "whale hunting ship captain obsessed with revenge",
]

ALL_SIZES = ["500", "2000", "5000"]


def api(base_url, path, params=None, timeout=600):
    url = base_url + path + (("?" + urllib.parse.urlencode(params)) if params else "")
    with urllib.request.urlopen(url, timeout=timeout) as res:
        return json.load(res)


def wait_for_backend(base_url, max_wait_s=180):
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            return api(base_url, "/api/health", timeout=5)
        except Exception:
            time.sleep(2)
    print(f"ERRORE: backend non raggiungibile entro {max_wait_s}s")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Benchmark di scalabilità della ricerca.")
    parser.add_argument("--base-url", default="http://localhost:5000")
    parser.add_argument("--sizes", nargs="*", default=ALL_SIZES, choices=ALL_SIZES)
    parser.add_argument("--reps", type=int, default=3, help="ripetizioni per query a caldo (default 3)")
    parser.add_argument("--max-queries", type=int, default=None, help="usa solo le prime N query")
    parser.add_argument("--restart", action="store_true",
                        help="riavvia gutenberg-backend prima di misurare (per la cache fredda)")
    parser.add_argument("--out", default="benchmark_results.csv")
    args = parser.parse_args()

    queries = QUERIES[:args.max_queries] if args.max_queries else QUERIES

    if args.restart:
        print("Riavvio del backend per misurare la cache fredda...")
        subprocess.run(["docker", "restart", "gutenberg-backend"], check=True, capture_output=True)

    health = wait_for_backend(args.base_url)
    print(f"Backend pronto. Stato cache: "
          f"{ {s: d['loaded'] for s, d in sorted(health['datasets'].items(), key=lambda kv: int(kv[0]))} }")

    raw_rows = []   # (size, phase, query, rep, time_s)
    summary  = []   # (size, docs, cold_time, cold_valid, warm_mean, warm_std, n_warm)

    for size in sorted(args.sizes, key=int):
        info = health["datasets"][size]
        if not info["documents"]:
            print(f"\nDataset {size}: non seedato, salto.")
            continue

        print(f"\n{'='*60}\nDataset {size} ({info['documents']} libri)\n{'='*60}")

        # ── Prima query: a cache fredda include il caricamento lazy ──────────
        cold_valid = not info["loaded"]
        data = api(args.base_url, "/api/search", {"q": queries[0], "top": 10, "dataset": size})
        cold_time = data["time_s"]
        raw_rows.append((size, "cold" if cold_valid else "already-warm", queries[0], 0, cold_time))
        label = "cache fredda (incl. caricamento)" if cold_valid else "già in cache (NON è un tempo a freddo)"
        print(f"Prima query [{label}]: {cold_time}s")

        # ── Batteria a caldo ─────────────────────────────────────────────────
        warm_times = []
        for q_idx, q in enumerate(queries, 1):
            for rep in range(1, args.reps + 1):
                data = api(args.base_url, "/api/search", {"q": q, "top": 10, "dataset": size})
                warm_times.append(data["time_s"])
                raw_rows.append((size, "warm", q, rep, data["time_s"]))
            print(f"  query {q_idx}/{len(queries)} completata", end="\r")
        print()

        warm_mean = statistics.mean(warm_times)
        warm_std  = statistics.stdev(warm_times) if len(warm_times) > 1 else 0.0
        summary.append((size, info["documents"], cold_time, cold_valid, warm_mean, warm_std, len(warm_times)))
        print(f"A caldo: media {warm_mean:.2f}s +/- {warm_std:.2f}s su {len(warm_times)} ricerche")

    # ── Riepilogo e CSV ──────────────────────────────────────────────────────
    print(f"\n{'='*60}\nRIEPILOGO\n{'='*60}")
    print("| Catalogo | Libri | Prima query (freddo) | Query a caldo (media +/- std) | N |")
    print("|---|---|---|---|---|")
    for size, docs, cold, cold_valid, mean, std, n in summary:
        cold_str = f"{cold}s" if cold_valid else f"{cold}s (*)"
        print(f"| {size} | {docs} | {cold_str} | {mean:.2f}s +/- {std:.2f}s | {n} |")
    if any(not s[3] for s in summary):
        print("(*) dataset già in cache al momento della misura: NON è un tempo a freddo.")
        print("    Rilancia con --restart per misurare il caricamento.")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "phase", "query", "rep", "time_s"])
        w.writerows(raw_rows)
    print(f"\nMisure grezze salvate in {args.out}")


if __name__ == "__main__":
    main()
