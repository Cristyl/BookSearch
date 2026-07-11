# test_api.py — smoke test dell'API di ricerca (solo stdlib, nessuna dipendenza).
#
# Uso:
#   python test_api.py                       # backend su http://localhost:5000
#   python test_api.py http://altro:5000     # base URL alternativo

import json
import sys
import urllib.parse
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
QUERY    = "a sea adventure involving a giant whale"


def get_json(url):
    with urllib.request.urlopen(url, timeout=300) as res:
        return json.load(res)


def main():
    print(f"Health check su {BASE_URL}/api/health ...")
    try:
        health = get_json(f"{BASE_URL}/api/health")
    except Exception as e:
        print(f"ERRORE: backend non raggiungibile ({e})")
        sys.exit(1)

    print(f"Status: {health['status']}")
    for size, info in sorted(health["datasets"].items(), key=lambda kv: int(kv[0])):
        print(f"  dataset {size:>5}: {info['documents']} documenti, "
              f"idf_model={'ok' if info['idf_model'] else 'MANCANTE'}, "
              f"loaded={info['loaded']}")

    failures = 0
    for size, info in sorted(health["datasets"].items(), key=lambda kv: int(kv[0])):
        if not info["documents"] or not info["idf_model"]:
            print(f"\nDataset {size}: non inizializzato, salto la ricerca "
                  f"(esegui 'python seed_datasets.py {size}').")
            continue

        print(f"\nRicerca su dataset {size}: \"{QUERY}\"")
        params = urllib.parse.urlencode({"q": QUERY, "top": 3, "dataset": size})
        try:
            data = get_json(f"{BASE_URL}/api/search?{params}")
        except Exception as e:
            print(f"  ERRORE: {e}")
            failures += 1
            continue

        print(f"  {data['count']} risultati in {data['time_s']}s")
        for r in data["results"]:
            print(f"    {r['rank']}. {r['title']} — {r['author']} ({r['similarity']})")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
