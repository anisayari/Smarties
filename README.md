# Smarties

Smarties classifies text by topic. Train it on your own labeled documents, or use
its Wikipedia tools to build a corpus from a list of topics. It runs on CPU.

## Install

The 0.12 series requires Python 3.10 or newer. To install this release from GitHub:

```sh
python -m pip install "Smarties @ git+https://github.com/anisayari/Smarties.git@v0.12.0"
```

For development, clone this repository and run `python -m pip install -e .`.
The default classifier works offline with the base installation.

## Train and predict

```python
from Smarties import TextClassifier

model = TextClassifier().fit(
    ["rocket in lunar orbit", "planet and moon", "team scores a goal", "football team wins"],
    ["space", "space", "sport", "sport"],
)
documents = ["a rocket enters orbit", "the football team scores"]
print(model.predict(documents))         # ['space' 'sport']
print(model.classes_)                   # column order in predict_proba
print(model.predict_proba(documents))   # two rows, one per document
```

Pass a list of strings, including for a single prediction. You can use integer
or string labels. Fit once, then reuse the fitted vocabulary, encoder and class
order across batches. To try the five sports in the bundled Wikipedia corpus,
run `python example/main.py "the football team won a soccer match"` after installation
from a clone.

| Method | Features | When to try it |
| --- | --- | --- |
| `tfidf` (default) | Word unigrams and bigrams, plus character n-grams of length 3–5 | A first model for a new domain, especially when spelling, abbreviations or proper names carry information. |
| `embedding` | Normalized document vectors from a local Model2Vec model | When a compact semantic representation is useful and an encoder exists for your language. |
| `hybrid` | Sparse lexical features and the document vector together | When you want to retain exact terms and also compare documents by meaning. |

All three train a logistic regression on your labels. `max_features` sets the
vocabulary cap for each lexical feature set. `C` controls regularization.
Choose the method on held-out documents from your domain.
[Benchmark and per-class results](https://github.com/anisayari/Smarties/blob/v0.12.0/benchmarks/README.md).

### Use a local semantic encoder

In a clone, run `python -m pip install -e ".[semantic]"`. Download an encoder
once, then pass its directory to Smarties. The benchmark uses this pinned
English encoder. Choose a suitable model for other languages.

```sh
hf download minishlab/potion-base-8M config.json tokenizer.json model.safetensors --revision bf8b056651a2c21b8d2565580b8569da283cab23 --local-dir .cache/potion-base-8M
```

```python
model = TextClassifier(method="hybrid", encoder=".cache/potion-base-8M")
model.fit(training_texts, training_labels)
labels = model.predict(new_texts)
```

[Model2Vec](https://github.com/MinishLab/model2vec) · [Encoder model card](https://huggingface.co/minishlab/potion-base-8M).

## Build a Wikipedia corpus

The network step uses the MediaWiki API. A seed pair contains the article title
and the label to learn. With `find_links=True`, Smarties also looks for articles
that link back to the seed. `max_article_links` limits the related articles
kept per topic. Run this example in a writable working directory.

```python
import Smarties as sm

sm.set_lang("english")
sm.construct_wiki_dico(
    "wiki.json", [("Association football", "sport"), ("Astronomy", "space")],
    init=True, find_links=False,
)
sm.construct_database_from_knowledge_base("wiki.json", "articles.csv")
articles = sm.import_database("articles.csv", split=False)
model = sm.TextClassifier().fit(articles["Content"], articles["Class"])
```

The CSV has `Content`, `Class`, `SubClass` (the article title) and
`class_labelized` (the numeric label). The exporter also writes `mapping.json`
in the working directory. Train/test splits should keep an entire article on
one side, including when training on its sentences.

`init=True` rebuilds the topics listed in the call. The default merges new
articles into existing topics. When applying a link cap to an old dictionary,
pass all seed titles you want to keep for a topic. The original JSON format
stores only article titles and IDs. Smarties removes aliases that resolve to
the same article and rejects an article explicitly seeded under two labels.

## Updating from 0.11

* Existing semicolon-delimited corpora and Wikipedia JSON dictionaries are
  supported. Both spellings of `construct_database_from_knowledge_base` and
  `construct_database_from_knwoledge_base` work.
* `model_from_database(df)` returns a fitted `TextClassifier` and a DataFrame.
  The classifier now takes raw text. `predict(clf, df, sentence)` still prints
  the class probabilities and also returns the predicted label. Retrain from
  the original text when moving an old Doc2Vec/random forest model to 0.12.
  If your code reads `vectorized_comments`, call the low-level Doc2Vec
  helpers explicitly to produce that column.
* English RAKE keyword extraction works with the base installation. Install
  `.[legacy]` in a clone for additional stopword lists, the low-level Gensim
  Doc2Vec helpers, Wikimedia page views and NetworkX graphs. You can supply
  a UTF-8 stopword file with `stoppath`. `get_graph_links` returns a graph in
  memory.
* `get_page` provides the article title, ID, URL, summary, content and links.
  Article text and links are fetched on demand. Supply an unambiguous title
  when building a corpus.

## License

See the [GNU General Public License, version 3](https://github.com/anisayari/Smarties/blob/master/LICENSE).
The original project includes the following author notice:

> You must report each utilisation of this code to the author of this code (anisayari).
> Please push your code using this API on a forked Github repo public.
