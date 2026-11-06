# Smarties classification benchmark

The benchmark compares three Smarties classifiers and three reference models on labeled documents. Each method sees the same train/test partitions. The optional encoder runs on CPU.

## Results from November 6, 2026

On the full 20 Newsgroups split, the hybrid method classified **5,197 of 7,280** held-out posts correctly. The default TF-IDF method classified **5,166** correctly and required about a quarter of the serialized storage (11.18 vs. 42.16 MB). Embeddings alone scored highest on the small sports collection (24 of 27 articles).

| Model | Sports macro-F1 | Balanced 20NG macro-F1 | Full 20NG macro-F1 | Full 20NG accuracy | Full fit, s | Full batch, ms/doc | Full model, MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0.093 | 0.005 | 0.005 | 5.27% | 0.004 | <0.001 | 0.002 |
| Unigram | 0.602 | 0.586 | 0.675 | 69.22% | 16.47 | 0.112 | 18.42 |
| Doc2Vec + RF | 0.384 | 0.210 | 0.366 | 38.68% | 45.92 | 2.302 | 198.25 |
| Smarties TF-IDF | 0.746 | 0.643 | 0.697 | 70.96% | 49.28 | 1.602 | 11.18 |
| Smarties embedding | **0.886** | 0.607 | 0.634 | 64.70% | 8.02 | 0.500 | 30.99 |
| Smarties hybrid | 0.817 | **0.647** | **0.701** | **71.39%** | 66.42 | 2.104 | 42.16 |

Topic boundaries account for some errors. With the full training set, the hybrid model labeled 80 `talk.politics.misc` posts as `talk.politics.guns`. Its weakest class was `talk.religion.misc` (F1 0.349 across 244 test posts). The class confusion counts show where aggregate accuracy hides these differences. [Per-class precision, recall and F1](results/per_class.csv) · [Confusion counts](results/confusion.csv)

The full 20 Newsgroups evaluation used 10,961 training posts and 7,280 test posts after cleaning. The balanced evaluation used 2,000 and 1,000 respectively. Two independent processes reproduced identical folds, confusion counts and quality metrics for all six methods on sports and the balanced sample. For all 18 evaluations, we also recomputed accuracy, per-class metrics and macro-F1 from the confusion counts. [Full precision and timings](results/summary.csv) · [Per-fold results](results/folds.csv)

## Reproduce

From the repository root, install Python 3.12 and the optional model dependencies. The run below pins the model and numerical stack used for the [results](results/environment.json). The first 20 Newsgroups run downloads the dataset into scikit-learn's local data cache.

```sh
python -m pip install -c benchmarks/constraints.txt -e ".[legacy,semantic]"
python -c "from huggingface_hub import snapshot_download; snapshot_download('minishlab/potion-base-8M', revision='bf8b056651a2c21b8d2565580b8569da283cab23', allow_patterns=['config.json','tokenizer.json','model.safetensors'], local_dir='.cache/potion-base-8M')"
PYTHONHASHSEED=42 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m benchmarks.run
```

In PowerShell, set the four environment variables with `$env:NAME='value'` before calling `python -m benchmarks.run`. For a quick offline run after installing the core package, use `python -m benchmarks.run --datasets sports --methods majority unigram tfidf --output .cache/bench-smoke`. The full run writes aggregate, per-fold and per-class metrics and nonzero confusion counts to [results](results/).

## What the splits measure

- **Sports.** The repository's Wikipedia snapshot has 28 rows. Two soccer titles refer to the same page with identical text. Deduplication leaves 27 articles across five sports. The benchmark classifies the full article body and fits each model five times with stratified, shuffled folds (seed 42). Each retained article contributes exactly one held-out prediction. The summary pools these predictions before computing macro-F1. An error moves accuracy by 3.7 percentage points. [Source corpus](../example/database_file_custom_name.csv)
- **20 Newsgroups.** We use all 20 categories and the official chronological train/test partitions, strip headers, footers and quoted material, and discard entries without word characters. We visit each partition in seeded shuffled order and discard duplicates after case and whitespace normalization. The full evaluation keeps all remaining posts. The balanced evaluation takes 100 posts per class for training and 50 for testing. A test post matching any earlier training post is excluded even when the train quota omitted that post. Near duplicates and remnants of quoting may still occur. The balanced sample is a subset of the full evaluation, so these scores are related. [scikit-learn dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)
- Each vectorizer, Doc2Vec model and supervised classifier fits on the training side of its split. The pretrained English encoder is fixed. It could have encountered public texts during upstream training. Both corpora are English. The results do not establish performance on other languages or domains.

## Models

| Method | Features and classifier |
|---|---|
| Majority | Always predicts the most frequent training label. Ties follow scikit-learn's class order. |
| Unigram | Default word TF-IDF and logistic regression (C=1). |
| Doc2Vec + RF | Gensim PV-DM with 100 dimensions, 10 epochs, minimum count 2 and one worker. A random forest with 100 trees learns from training document vectors. Prediction infers new vectors with the fitted encoder. |
| Smarties TF-IDF | Word unigrams/bigrams and character-boundary 3–5-grams, capped at 50,000 features each, with sublinear term frequencies. |
| Smarties embedding | Frozen 256-dimensional Model2Vec features, normalized per document. |
| Smarties hybrid | Concatenated TF-IDF and normalized Model2Vec features. |

All three Smarties variants use SAGA logistic regression with C=4 and seed 42. Model2Vec 0.10 and this encoder revision read up to 512 tokens per document. The TF-IDF branches see the full text. The historical-family baseline uses fixed settings to measure Doc2Vec + RF on modern dependencies. Version 0.11 trained a fresh document embedding space during prediction. We do not attribute this reference implementation's score or speed to that release. [Original prediction code](https://github.com/anisayari/Smarties/blob/064f521/Smarties/SmartiesAPI.py) · [Encoder revision](https://huggingface.co/minishlab/potion-base-8M/tree/bf8b056651a2c21b8d2565580b8569da283cab23) · [Gensim Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html)

## Timing and size

Fit time includes local model loading for the semantic variants and excludes dataset preparation and downloads. Batch latency is total held-out prediction time divided by held-out document count. Each model also predicts one document per call after the batch warm-up. The script uses all 27 held-out sports articles and a seeded sample of 50 posts per newsgroup evaluation for single-call p50/p95. Serialized size counts uncompressed Python protocol-5 bytes for the fitted estimator, including the stored encoder when applicable. Fit time and size are medians over the five sports fits or the single fit per newsgroup evaluation. These are measurements from one run on a 20-core NVIDIA N1X Windows ARM64 system running x64 Python under emulation. Native BLAS/OpenMP threads were limited to one and tokenizer parallelism was disabled. Timing varies with hardware and host load.
