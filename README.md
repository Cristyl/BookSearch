# BookSearch — Ricerca di libri Gutenberg per trama

Motore di ricerca "search by plot": descrivi una trama e il sistema restituisce i libri
del Project Gutenberg più simili, ordinati per cosine similarity su vettori TF-IDF
calcolati con PySpark e salvati su MongoDB.

## Architettura

```
                          ┌───────────────────────────────┐
  (una tantum, in locale) │  seed_datasets.py             │
  CSV in data/  ─────────▶│  pipeline NLP + TF-IDF Spark  │
                          └──────────┬────────────────────┘
                                     │ scrive
                                     ▼
   ┌─────────────┐   ┌────────────────────────────────┐   ┌──────────────┐
   │  frontend   │──▶│  backend (Flask + Spark)       │──▶│  MongoDB     │
   │  nginx :80  │   │  :5000                         │   │  :27017      │
   │  selettore  │   │  carica lazy il dataset scelto │   │  books_tfidf │
   │ 500/2000/   │   │  + modello IDF da ./idf_model/ │   │  _500/_2000/ │
   │ 5000        │   │                                │   │  _5000       │
   └─────────────┘   └────────────────────────────────┘   └──────────────┘
```

- Il calcolo TF-IDF **non avviene più all'avvio di Docker**: si fa una volta sola in
  locale con `seed_datasets.py`. I dati restano nel volume Docker `mongo-data`.
- Ogni dataset ha il **suo** modello IDF in `idf_model/{500,2000,5000}` (l'IDF dipende
  dal corpus, non sono intercambiabili). La cartella è montata read-only nel backend.
- Il frontend sceglie il catalogo (500/2000/5000) e il backend interroga la collection
  corrispondente (`books_tfidf_{size}`), caricandola in cache Spark al primo uso.

## Prerequisiti

Per **avviare** l'applicazione: solo Docker Desktop.

Per il **seeding** (una tantum, in locale su Windows):
- Python 3.11 con: `pyspark==3.5.3`, `pandas`, `pymongo`, `nltk`, `numpy`
- JDK 11 (default: `C:\Program Files\Eclipse Adoptium\jdk-11.0.31.11-hotspot`)
- Hadoop winutils in `C:\hadoop` (percorsi diversi? Imposta `JAVA_HOME` / `HADOOP_HOME`
  come variabili d'ambiente prima di lanciare lo script)
- I CSV puliti in `data/gutenberg_data_{500,2000,5000}_clean.csv`

## Primo avvio

```powershell
# 1. Avvia solo MongoDB (crea il volume persistente)
docker compose up -d mongodb

# 2. Seeding: calcola TF-IDF e riempie le tre collection (LENTO, soprattutto la 5000)
python seed_datasets.py            # tutte e tre
#   oppure una alla volta:
python seed_datasets.py 500
python seed_datasets.py 2000
python seed_datasets.py 5000

# 3. Avvia tutto il resto
docker compose up -d --build

# 4. Verifica
python test_api.py
```

Poi apri http://localhost. Nota: la **prima ricerca** su un catalogo non ancora caricato
è lenta (il backend carica i vettori in cache Spark); le successive sono veloci.
Il catalogo 500 viene precaricato all'avvio (`PRELOAD_DATASETS` in `docker-compose.yml`).

## Avvii successivi

```powershell
docker compose up -d      # nessun ricalcolo: i dati sono già nel volume
docker compose stop       # ferma tutto senza perdere nulla
```

`docker compose down` è sicuro (il volume `mongo-data` sopravvive).
**MAI `docker compose down -v`**: cancella il volume e costringe a rifare tutto il seeding.

## Cosa rifare quando cambio qualcosa

| Cosa ho cambiato                          | Cosa devo rifare                                                        |
|-------------------------------------------|-------------------------------------------------------------------------|
| CSV di **una** size (es. 2000)            | `python seed_datasets.py 2000` poi `docker compose restart backend`     |
| Codice backend (`backend/`)               | `docker compose up -d --build backend`                                  |
| Frontend (`frontend/index.html`)          | `docker compose up -d --build frontend`                                 |
| `NUM_FEATURES` o pipeline NLP nel seeder  | Ri-seeding di **tutte** le size + allineare `NUM_FEATURES` in compose + `docker compose up -d --build backend` |
| `docker-compose.yml` (env, porte...)      | `docker compose up -d`                                                  |
| Ho fatto `down -v` per errore             | Ripartire da "Primo avvio" (passo 1)                                    |

Il restart del backend dopo un ri-seeding serve perché il backend tiene i vettori
in cache Spark: senza restart continuerebbe a servire i dati vecchi.

## API

- `GET /api/health` — stato + per ogni dataset: numero documenti, modello IDF presente,
  caricato in cache.
- `GET /api/search?q=<trama>&dataset=<500|2000|5000>&top=<1-50>` — ricerca.
  `dataset` è opzionale (default 500). Risposta: `{query, dataset, count, time_s, results[]}`.

Errori utili:
- `400` — parametri mancanti/non validi.
- `503` — il dataset richiesto non è stato seedato: esegui `seed_datasets.py <size>`.

## Benchmark e test di qualità

Con lo stack attivo:

```powershell
python benchmark.py --restart    # tempi a cache fredda e calda per catalogo → benchmark_results.csv
python quality_test.py           # self-retrieval: Top-1/Top-5/MRR per catalogo → quality_results.csv
```

`benchmark.py --restart` riavvia il backend per misurare anche il caricamento lazy.
`quality_test.py` legge i CSV in `data/` (servono in locale) e usa estratti dei libri
come query, verificando che ogni libro ritrovi sé stesso. Opzioni: `--sample`,
`--lengths`, `--sizes`, `--reps` (vedi `--help`).

## Condividere il progetto con il database già pronto

Chi riceve il progetto NON deve rifare il seeding né avere i CSV/Spark in locale:
bastano Docker, il codice, la cartella `idf_model/` e un dump del database.

**Chi ha i dati (esporta):**

```powershell
docker exec gutenberg-mongo mongodump --db gutenberg --gzip --archive=/tmp/gutenberg.dump.gz
docker cp gutenberg-mongo:/tmp/gutenberg.dump.gz .\gutenberg.dump.gz
```

Da condividere: il progetto (senza i CSV, non servono), **inclusa** `idf_model/`,
più il file `gutenberg.dump.gz`.

**Chi riceve (importa):**

```powershell
docker compose up -d mongodb
docker cp .\gutenberg.dump.gz gutenberg-mongo:/tmp/gutenberg.dump.gz
docker exec gutenberg-mongo mongorestore --gzip --archive=/tmp/gutenberg.dump.gz --drop
docker compose up -d --build
python test_api.py   # verifica (opzionale)
```

I modelli IDF e le collection devono provenire **dallo stesso seeding** (vedi
"Vincoli da non rompere"): condividi sempre dump e `idf_model/` insieme.

## Attenzione: MongoDB nativo su Windows

Se sulla macchina gira anche un **MongoDB installato nativamente** (servizio Windows),
occupa `127.0.0.1:27017` e tutto ciò che punta a `localhost:27017` (il seeder, Compass)
finisce **lì** invece che nel Mongo del container — che resta vuoto mentre "in Compass
i dati ci sono". Prima di fare seeding, ferma il servizio nativo:

```powershell
Stop-Service MongoDB        # poi riavvia il container: docker compose restart mongodb
```

In alternativa lancia il seeder puntando esplicitamente al container via IPv6:
`$env:MONGO_URI = "mongodb://[::1]:27017/"; python seed_datasets.py ...`

## Vincoli da non rompere

- `NUM_FEATURES = 100000` deve essere **identico** in `seed_datasets.py` e nel backend
  (env `NUM_FEATURES` in `docker-compose.yml`): valori diversi producono similarità
  silenziosamente sbagliate, senza errori.
- La pipeline di preprocessing della query nel backend (regex `[^a-z\s]`, stopwords
  inglesi, `SnowballStemmer("english")`) deve restare identica a quella del seeder.
- Ogni collection `books_tfidf_{size}` è valida solo con il modello `idf_model/{size}`
  generato dallo **stesso** run di seeding: rigenerarne uno solo dei due li disallinea.
