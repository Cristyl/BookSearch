# seed_datasets.py — calcola le collection TF-IDF in locale, UNA volta per dataset.
#
# Per ogni size (500, 2000, 5000):
#   - carica data/gutenberg_data_{size}_clean.csv nel MongoDB dockerizzato
#     (collezione temporanea books_raw_{size})
#   - esegue la pipeline NLP con Spark (pulizia -> tokenizzazione -> stopwords -> stemming)
#   - calcola TF-IDF e salva i vettori in books_tfidf_{size}
#   - salva il modello IDF in ./idf_model/{size} (montato poi nel backend)
#
# Uso:
#   python seed_datasets.py              # tutte le size
#   python seed_datasets.py 5000         # solo una size
#   python seed_datasets.py 500 2000     # solo alcune
#
# Prerequisito: MongoDB attivo su localhost:27017 (docker compose up -d mongodb)

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configurazione ambiente Hadoop / Java / Spark.
# setdefault: se le variabili sono gia' impostate nel sistema, non vengono sovrascritte.
os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Eclipse Adoptium\jdk-11.0.31.11-hotspot")
if r"C:\hadoop\bin" not in os.environ["PATH"]:
    os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import pandas as pd
from pymongo import MongoClient
from nltk.stem import SnowballStemmer

from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, lower, regexp_replace, trim, pandas_udf
from pyspark.sql.types import ArrayType, StringType

MONGO_URI      = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
JAR_PATH       = os.path.join(BASE_DIR, "jars", "mongo-spark-connector_2.12-10.3.0-all.jar")
IDF_BASE_PATH  = os.path.join(BASE_DIR, "idf_model")
NUM_FEATURES   = 100000  # DEVE combaciare con NUM_FEATURES del backend (docker-compose.yml)
ALL_SIZES      = ["500", "2000", "5000"]

CSV_CHUNK_ROWS = 200  # righe CSV per batch in ingresso (i testi pesano molto)
BATCH_SIZE_OUT = 200  # documenti TF-IDF per batch in uscita

stemmer = SnowballStemmer("english")


@pandas_udf(ArrayType(StringType()))
def stem_pandas_udf(series: pd.Series) -> pd.Series:
    """Stemming a blocchi (batch) sfruttando Apache Arrow."""
    return series.apply(lambda tokens: [stemmer.stem(t) for t in tokens if t] if tokens is not None else [])


def load_csv_to_mongo(csv_path, collection):
    """Carica il CSV nella collezione temporanea, a chunk per non saturare la RAM."""
    collection.drop()
    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=CSV_CHUNK_ROWS):
        chunk = chunk.dropna(subset=["Text", "Title"])
        docs = []
        for _, row in chunk.iterrows():
            docs.append({
                "title":     row["Title"],
                "author":    row["Author"]    if pd.notna(row.get("Author"))    else None,
                "link":      row["Link"]      if pd.notna(row.get("Link"))      else None,
                "book_id":   int(row["ID"])   if pd.notna(row.get("ID"))        else None,
                "bookshelf": row["Bookshelf"] if pd.notna(row.get("Bookshelf")) else None,
                "text":      row["Text"],
            })
        if docs:
            collection.insert_many(docs)
            total += len(docs)
            print(f"  ...caricati {total} libri", end="\r")
    print()
    return total


def seed_dataset(size, spark, db):
    csv_path = os.path.join(BASE_DIR, "data", f"gutenberg_data_{size}_clean.csv")
    if not os.path.isfile(csv_path):
        print(f"ERRORE: {csv_path} non trovato, size {size} saltata.")
        return False

    print(f"\n{'='*60}\nDataset {size}: caricamento CSV su MongoDB...\n{'='*60}")
    col_raw = db[f"books_raw_{size}"]
    total = load_csv_to_mongo(csv_path, col_raw)
    print(f"Caricati {total} libri in books_raw_{size}")

    df = spark.read.format("mongodb") \
        .option("spark.mongodb.read.connection.uri", MONGO_URI) \
        .option("spark.mongodb.read.database", "gutenberg") \
        .option("spark.mongodb.read.collection", f"books_raw_{size}") \
        .load()

    # Partizionamento dinamico per non strozzare i worker con i testi grandi
    num_partitions = 1000 if size == "5000" else 200
    df = df.repartition(num_partitions)

    # Pipeline NLP: pulizia -> tokenizzazione -> stopwords -> stemming
    df_clean = df.withColumn("text_clean", trim(regexp_replace(lower(col("text")), r"[^a-z\s]", "")))
    df_tok   = Tokenizer(inputCol="text_clean", outputCol="tokens").transform(df_clean)
    df_ns    = StopWordsRemover(inputCol="tokens", outputCol="tokens_no_stop").transform(df_tok)
    df_pre   = df_ns.withColumn("tokens_filtered", stem_pandas_udf("tokens_no_stop"))

    # TF-IDF
    hashingTF = HashingTF(inputCol="tokens_filtered", outputCol="tf_vector", numFeatures=NUM_FEATURES)
    df_tf     = hashingTF.transform(df_pre)
    idf_model = IDF(inputCol="tf_vector", outputCol="tfidf_vector").fit(df_tf)
    df_tfidf  = idf_model.transform(df_tf)

    idf_path = os.path.join(IDF_BASE_PATH, size)
    idf_model.write().overwrite().save(idf_path)
    print(f"Modello IDF salvato in {idf_path}")

    # Scrittura risultati: toLocalIterator per non far esplodere la RAM del driver
    col_tfidf = db[f"books_tfidf_{size}"]
    col_tfidf.drop()

    rows = df_tfidf.select("title", "author", "link", "book_id", "bookshelf", "tfidf_vector").toLocalIterator()
    docs, saved = [], 0
    for row in rows:
        vec = row["tfidf_vector"]
        docs.append({
            "title":         row["title"],
            "author":        row["author"],
            "link":          row["link"],
            "book_id":       row["book_id"],
            "bookshelf":     row["bookshelf"],
            "tfidf_indices": [int(i) for i in vec.indices],
            "tfidf_values":  [float(v) for v in vec.values],
        })
        if len(docs) == BATCH_SIZE_OUT:
            col_tfidf.insert_many(docs)
            saved += len(docs)
            print(f"  ...salvati {saved} vettori", end="\r")
            docs = []
    if docs:
        col_tfidf.insert_many(docs)
        saved += len(docs)
    print(f"\nSalvati {saved} vettori in books_tfidf_{size}")

    col_raw.drop()  # pulizia collezione temporanea
    return True


def main():
    parser = argparse.ArgumentParser(description="Seeding delle collection TF-IDF su MongoDB.")
    parser.add_argument("sizes", nargs="*", default=ALL_SIZES,
                        help=f"Size da processare tra {ALL_SIZES} (default: tutte)")
    args = parser.parse_args()
    sizes = args.sizes or ALL_SIZES
    invalid = [s for s in sizes if s not in ALL_SIZES]
    if invalid:
        parser.error(f"size non valide: {invalid} (ammesse: {ALL_SIZES})")

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception:
        print(f"ERRORE: MongoDB non raggiungibile su {MONGO_URI}")
        print("Avvialo prima con: docker compose up -d mongodb")
        sys.exit(1)
    db = client["gutenberg"]

    spark = SparkSession.builder \
        .appName("GutenbergSeed") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "100s") \
        .config("spark.jars", JAR_PATH) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    ok = [s for s in sizes if seed_dataset(s, spark, db)]

    spark.stop()
    client.close()
    print(f"\nSeeding completato per: {', '.join(ok) if ok else 'nessuna size'}")
    if set(ok) != set(sizes):
        sys.exit(1)


if __name__ == "__main__":
    main()
