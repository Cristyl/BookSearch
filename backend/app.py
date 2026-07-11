import os
import re
import time
import logging
import threading

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, StopWordsRemover, IDFModel
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType
from nltk.stem import SnowballStemmer

MONGO_URI      = os.environ.get("MONGO_URI", "mongodb://mongodb:27017/")
MONGO_DB       = os.environ.get("MONGO_DB", "gutenberg")
IDF_MODEL_PATH = os.environ.get("IDF_MODEL_PATH", "/app/idf_model")
NUM_FEATURES   = int(os.environ.get("NUM_FEATURES", 100000))  # DEVE combaciare con seed_datasets.py
JAR_PATH       = "/app/jars/mongo-spark-connector_2.12-10.3.0-all.jar"

DATASET_SIZES   = ("500", "2000", "5000")
DEFAULT_DATASET = os.environ.get("DEFAULT_DATASET", "500")
# Dataset da precaricare all'avvio (lista separata da virgole, vuota = nessuno)
PRELOAD_DATASETS = [s.strip() for s in os.environ.get("PRELOAD_DATASETS", "").split(",") if s.strip()]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gutenberg-search-api")

log.info("Avvio SparkSession...")
spark = SparkSession.builder \
    .appName("GutenbergSearchAPI") \
    .config("spark.driver.memory", "4g") \
    .config("spark.jars", JAR_PATH) \
    .config("spark.mongodb.read.connection.uri", MONGO_URI) \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

mongo_client = MongoClient(MONGO_URI)

stopwords = StopWordsRemover.loadDefaultStopWords("english")
stemmer   = SnowballStemmer("english")
hashingTF = HashingTF(inputCol="tokens_filtered", outputCol="tf_vector", numFeatures=NUM_FEATURES)

# Cache dei dataset caricati: size -> {"df", "idf", "count"}
_datasets = {}
_load_lock = threading.Lock()


def collection_name(size):
    return f"books_tfidf_{size}"


def get_dataset(size):
    """Carica (lazy) il DataFrame TF-IDF e il modello IDF di una size, con cache."""
    if size in _datasets:
        return _datasets[size]

    with _load_lock:
        if size in _datasets:
            return _datasets[size]

        if mongo_client[MONGO_DB][collection_name(size)].estimated_document_count() == 0:
            raise LookupError(
                f"La collection '{collection_name(size)}' è vuota o assente: "
                f"esegui in locale 'python seed_datasets.py {size}' e riprova."
            )

        model_path = os.path.join(IDF_MODEL_PATH, size)
        if not os.path.isdir(model_path):
            raise LookupError(
                f"Modello IDF mancante in {model_path}: "
                f"esegui in locale 'python seed_datasets.py {size}' e riavvia il backend."
            )

        log.info(f"Carico dataset {size} da MongoDB...")
        # L'URI va ripetuto nelle opzioni di lettura: quando si passano opzioni,
        # il connettore v10 ignora la config di sessione e ripiega su localhost
        df = spark.read.format("mongodb") \
            .option("spark.mongodb.read.connection.uri", MONGO_URI) \
            .option("spark.mongodb.read.database", MONGO_DB) \
            .option("spark.mongodb.read.collection", collection_name(size)) \
            .load()
        # 'link' è presente solo nelle collection ri-seedate: selezioniamo ciò che c'è
        cols = [c for c in ("title", "author", "link", "tfidf_indices", "tfidf_values") if c in df.columns]
        df = df.select(*cols).cache()
        count = df.count()

        idf_model = IDFModel.load(model_path)
        _datasets[size] = {"df": df, "idf": idf_model, "count": count}
        log.info(f"Dataset {size} pronto: {count} libri.")
        return _datasets[size]


def make_similarity_udf(query_broadcast, norm_query):
    def cosine_similarity(indices, values):
        if not indices or not values:
            return 0.0
        query_vec = query_broadcast.value
        dot = sum(query_vec.get(int(i), 0.0) * float(v)
                  for i, v in zip(indices, values) if i is not None and v is not None)
        if dot == 0.0:
            return 0.0
        norm_book = sum(float(v) ** 2 for v in values if v is not None) ** 0.5
        if norm_book == 0.0 or norm_query == 0.0:
            return 0.0
        return float(dot / (norm_book * norm_query))
    return udf(cosine_similarity, FloatType())


def search_books(raw_query, size, top_n=10):
    query_clean   = re.sub(r"[^a-z\s]", "", raw_query.lower()).strip()
    query_tokens  = query_clean.split()
    query_no_stop = [w for w in query_tokens if w not in stopwords]
    query_stemmed = [stemmer.stem(w) for w in query_no_stop]

    if not query_stemmed:
        return []

    dataset = get_dataset(size)

    df_query = spark.createDataFrame([(query_stemmed,)], ["tokens_filtered"])
    df_query = hashingTF.transform(df_query)
    df_query = dataset["idf"].transform(df_query)

    query_vector = df_query.select("tfidf_vector").collect()[0]["tfidf_vector"]
    query_dict   = {int(i): float(v) for i, v in zip(query_vector.indices, query_vector.values)}
    norm_query   = sum(v ** 2 for v in query_dict.values()) ** 0.5

    query_broadcast = spark.sparkContext.broadcast(query_dict)
    try:
        similarity_udf = make_similarity_udf(query_broadcast, norm_query)
        meta_cols = [c for c in ("title", "author", "link") if c in dataset["df"].columns]
        rows = dataset["df"] \
            .withColumn("similarity", similarity_udf(col("tfidf_indices"), col("tfidf_values"))) \
            .select(*meta_cols, "similarity") \
            .orderBy(col("similarity").desc()) \
            .limit(top_n) \
            .collect()
    finally:
        query_broadcast.destroy()

    return [
        {
            "rank":       i,
            "title":      row["title"],
            "author":     row["author"],
            "link":       row.asDict().get("link"),
            "similarity": round(float(row["similarity"]), 4),
        }
        for i, row in enumerate(rows, start=1)
    ]


app = Flask(__name__)
CORS(app)


@app.get("/api/health")
def health():
    datasets = {}
    for size in DATASET_SIZES:
        try:
            documents = mongo_client[MONGO_DB][collection_name(size)].estimated_document_count()
        except Exception:
            documents = None
        datasets[size] = {
            "documents": documents,
            "loaded":    size in _datasets,
            "idf_model": os.path.isdir(os.path.join(IDF_MODEL_PATH, size)),
        }
    return jsonify({"status": "ok", "default_dataset": DEFAULT_DATASET, "datasets": datasets})


@app.get("/api/search")
def api_search():
    query   = request.args.get("q", "").strip()
    top_n   = request.args.get("top", default=10, type=int)
    top_n   = max(1, min(top_n, 50))
    dataset = request.args.get("dataset", DEFAULT_DATASET).strip()

    if not query:
        return jsonify({"error": "Parametro 'q' mancante o vuoto."}), 400
    if dataset not in DATASET_SIZES:
        return jsonify({"error": f"Parametro 'dataset' non valido: '{dataset}'. Ammessi: {list(DATASET_SIZES)}."}), 400

    try:
        start   = time.time()
        results = search_books(query, dataset, top_n)
        elapsed = round(time.time() - start, 2)
    except LookupError as e:
        log.warning(str(e))
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        log.exception("Errore durante la ricerca")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "query":   query,
        "dataset": dataset,
        "count":   len(results),
        "time_s":  elapsed,
        "results": results
    })


def preload():
    for size in PRELOAD_DATASETS:
        if size not in DATASET_SIZES:
            log.warning(f"PRELOAD_DATASETS contiene una size non valida: {size}")
            continue
        try:
            get_dataset(size)
        except Exception as e:
            log.warning(f"Preload dataset {size} fallito: {e}")


if __name__ == "__main__":
    threading.Thread(target=preload, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
