"""Reproduce Smarties' document-level classification and CPU cost measurements."""

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import random
import re
import sysconfig
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
METHODS = ("majority", "unigram", "doc2vec_rf", "tfidf", "embedding", "hybrid")
TOKEN = re.compile(r"\w+")


def fingerprint(text):
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class Corpus:
    texts: list
    labels: list
    splits: list
    notes: dict

    def split_hash(self):
        """Identify content, labels, order, and fold membership without storing the corpus."""
        payload = {
            "documents": [(fingerprint(text), label) for text, label in zip(self.texts, self.labels)],
            "splits": [(train.tolist(), test.tolist()) for train, test in self.splits],
        }
        return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def sports():
    texts, labels, seen = [], [], {}
    path = ROOT / "example" / "database_file_custom_name.csv"
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source, delimiter=";"))
    for row in rows:
        key = fingerprint(row["Content"])
        if key in seen:
            if seen[key] != row["Class"]:
                raise ValueError("A sports article has conflicting labels")
            continue
        seen[key] = row["Class"]
        texts.append(row["Content"])
        labels.append(row["Class"])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return Corpus(texts, labels, list(folds.split(texts, labels)), {"source_rows": len(rows)})


def newsgroups(full=False):
    texts, labels, seen = [], [], set()
    split = {}
    counts = {}
    for subset, limit in (("train", 100), ("test", 50)):
        data = fetch_20newsgroups(
            subset=subset, remove=("headers", "footers", "quotes"), shuffle=False
        )
        order = list(range(len(data.data)))
        random.Random(SEED).shuffle(order)
        indices = []
        class_counts = [0] * len(data.target_names)
        skipped_empty = skipped_duplicate = 0
        for index in order:
            text, label = data.data[index], data.target[index]
            if not TOKEN.search(text):
                skipped_empty += 1
                continue
            key = fingerprint(text)
            if key in seen:
                skipped_duplicate += 1
                continue
            # Test documents must be distinct from the entire original train partition.
            seen.add(key)
            if full or class_counts[label] < limit:
                class_counts[label] += 1
                indices.append(len(texts))
                texts.append(text)
                labels.append(data.target_names[label])
        if not full and min(class_counts) != limit:
            raise ValueError(f"Insufficient unique {subset} documents for the requested quota")
        counts[subset] = {
            "source_rows": len(order), "empty": skipped_empty, "duplicate": skipped_duplicate,
            "used_class_counts": dict(zip(data.target_names, class_counts)),
        }
        split[subset] = np.array(indices)
    return Corpus(texts, labels, [(split["train"], split["test"])], counts)


class Doc2VecForest:
    """Reference implementation of the historical model family with held-out inference."""

    def fit(self, texts, labels):
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        docs = [TaggedDocument(TOKEN.findall(text.lower()), [i]) for i, text in enumerate(texts)]
        self.encoder = Doc2Vec(
            documents=docs, vector_size=100, min_count=2, epochs=10, seed=SEED, workers=1
        )
        self.forest = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=SEED)
        self.forest.fit(self.encoder.dv.vectors, labels)
        return self

    def predict(self, texts):
        vectors = [self.encoder.infer_vector(TOKEN.findall(text.lower()), epochs=10) for text in texts]
        return self.forest.predict(vectors)


def new_model(method, encoder):
    if method == "majority":
        return DummyClassifier(strategy="most_frequent")
    if method == "unigram":
        return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000, random_state=SEED))
    if method == "doc2vec_rf":
        return Doc2VecForest()
    from Smarties.classifier import TextClassifier

    if method in ("embedding", "hybrid"):
        return TextClassifier(method=method, encoder=str(encoder))
    return TextClassifier(method=method)


class ByteCounter:
    """Count the uncompressed protocol-5 representation without retaining a second copy."""

    def __init__(self):
        self.size = 0

    def write(self, data):
        self.size += memoryview(data).nbytes


def evaluate(name, corpus, method, encoder):
    actual, predicted, single_ms, fits, sizes, fold_rows = [], [], [], [], [], []
    batch_seconds = 0.0
    categories = sorted(set(corpus.labels))
    for number, (train, test) in enumerate(corpus.splits, start=1):
        x_train = [corpus.texts[i] for i in train]
        y_train = [corpus.labels[i] for i in train]
        x_test = [corpus.texts[i] for i in test]
        y_test = [corpus.labels[i] for i in test]
        model = new_model(method, encoder)
        start = perf_counter()
        model.fit(x_train, y_train)
        fits.append(perf_counter() - start)
        start = perf_counter()
        y_pred = list(model.predict(x_test))
        batch_seconds += perf_counter() - start
        actual.extend(y_test)
        predicted.extend(y_pred)
        # The preceding batch warms up the model. Use the same documents for every method.
        for idx in random.Random(SEED).sample(range(len(x_test)), min(50, len(x_test))):
            start = perf_counter()
            model.predict([x_test[idx]])
            single_ms.append((perf_counter() - start) * 1000)
        counter = ByteCounter()
        pickle.dump(model, counter, protocol=5)
        sizes.append(counter.size)
        fold_rows.append({
            "dataset": name, "model": method, "fold": number,
            "train": len(train), "test": len(test), "accuracy": round(accuracy_score(y_test, y_pred), 6),
            "macro_f1": round(precision_recall_fscore_support(
                y_test, y_pred, labels=categories, average="macro", zero_division=0
            )[2], 6),
        })

    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=categories, zero_division=0
    )
    summary = {
        "dataset": name, "model": method, "held_out": len(actual), "folds": len(fits),
        "accuracy": round(accuracy_score(actual, predicted), 6), "macro_f1": round(float(f1.mean()), 6),
        "fit_s_median": round(median(fits), 5), "batch_ms_per_doc": round(1000 * batch_seconds / len(actual), 5),
        "single_p50_ms": round(median(single_ms), 5), "single_p95_ms": round(np.percentile(single_ms, 95), 5),
        "serialized_bytes_median": int(median(sizes)),
    }
    class_rows = [
        {"dataset": name, "model": method, "class": cat, "n": int(n),
         "precision": round(p, 6), "recall": round(r, 6), "f1": round(f, 6)}
        for cat, n, p, r, f in zip(categories, support, precision, recall, f1)
    ]
    matrix = confusion_matrix(actual, predicted, labels=categories)
    confusion = [
        {"dataset": name, "model": method, "actual": categories[row], "predicted": categories[col],
         "count": int(matrix[row, col])} for row, col in zip(*np.nonzero(matrix))
    ]
    return {"summary": [summary], "per_class": class_rows, "confusion": confusion, "folds": fold_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    datasets = ["sports", "newsgroups", "newsgroups_full"]
    parser.add_argument("--datasets", nargs="+", choices=datasets, default=datasets)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--encoder", type=Path, default=ROOT / ".cache" / "potion-base-8M")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results")
    args = parser.parse_args()
    if "doc2vec_rf" in args.methods and os.getenv("PYTHONHASHSEED") != str(SEED):
        parser.error(f"Set PYTHONHASHSEED={SEED} before starting Python for repeatable Doc2Vec inference")
    if {"hybrid", "embedding"}.intersection(args.methods) and not args.encoder.is_dir():
        parser.error("Provide --encoder pointing to the downloaded Model2Vec directory")
    args.output.mkdir(parents=True, exist_ok=True)
    packages = ["numpy", "scipy", "scikit-learn"]
    if "doc2vec_rf" in args.methods:
        packages.append("gensim")
    if {"hybrid", "embedding"}.intersection(args.methods):
        packages.extend(["model2vec", "tokenizers"])
    meta = {
        "python": platform.python_version(), "os": platform.system(), "os_version": platform.version(),
        "host_reported_machine": platform.machine(), "python_platform": sysconfig.get_platform(),
        "versions": {k: version(k) for k in packages},
        "thread_settings": {
            key: os.getenv(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM")
        },
        "threadpoolctl_limit": 1, "doc2vec_workers": 1, "forest_n_jobs": 1,
        "seed": SEED, "datasets": {},
    }
    if {"hybrid", "embedding"}.intersection(args.methods):
        meta["encoder_sha256"] = {
            file: hashlib.sha256((args.encoder / file).read_bytes()).hexdigest()
            for file in ("config.json", "tokenizer.json", "model.safetensors")
        }
    records = {"summary": [], "per_class": [], "confusion": [], "folds": []}
    for name in args.datasets:
        corpus = sports() if name == "sports" else newsgroups(full=name == "newsgroups_full")
        meta["datasets"][name] = {"sha256": corpus.split_hash(), **corpus.notes}
        print(f"{name}: {len(corpus.texts)} unique documents", flush=True)
        for method in args.methods:
            with threadpool_limits(limits=1):
                result = evaluate(name, corpus, method, args.encoder)
            print(result["summary"][0], flush=True)
            for key, rows in result.items():
                records[key].extend(rows)
    for key, rows in records.items():
        with (args.output / (key + ".csv")).open("w", encoding="utf-8", newline="") as dest:
            writer = csv.DictWriter(dest, fieldnames=rows[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "environment.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
