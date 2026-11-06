"""Train and reuse a supervised classifier for raw text documents."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_consistent_length, check_is_fitted


def _documents(texts):
    if isinstance(texts, (str, bytes, Mapping)) or getattr(texts, "ndim", 1) != 1:
        raise TypeError("Expected a collection of documents. For one document, use [text].")
    documents = list(texts)
    if not documents:
        raise ValueError("At least one document is required.")
    for index, document in enumerate(documents):
        if not isinstance(document, str) or not document.strip():
            raise ValueError(f"Document {index} must be a non-empty string.")
    return documents


class _StaticEmbeddings(TransformerMixin, BaseEstimator):
    """Pool token embeddings from a user-selected, local Model2Vec directory."""

    def __init__(self, encoder):
        self.encoder = encoder

    def fit(self, X, y=None):
        if self.encoder is None or not Path(self.encoder).is_dir():
            raise ValueError("Set encoder to the local directory of a Model2Vec model.")
        try:
            from model2vec import StaticModel
        except ImportError as exc:
            raise ImportError("Install Smarties[semantic] to use embeddings.") from exc
        self.model_ = StaticModel.from_pretrained(Path(self.encoder).resolve())
        return self

    def transform(self, X):
        check_is_fitted(self, "model_")
        vectors = self.model_.encode(X, use_multiprocessing=False)
        return normalize(np.asarray(vectors, dtype=np.float32))


class TextClassifier(ClassifierMixin, BaseEstimator):
    """Classify a sequence of text documents with a fitted logistic regression.

    method="tfidf" uses word and character n-grams. "embedding" uses a local
    Model2Vec encoder. "hybrid" concatenates all three feature sets. The
    encoder directory is read once during fit. max_features caps each TF-IDF
    vocabulary separately, and C controls inverse regularization strength.
    The columns of predict_proba follow classes_.
    """

    def __init__(self, method="tfidf", *, encoder=None, max_features=50000, C=4.0,
                 random_state=42):
        self.method = method
        self.encoder = encoder
        self.max_features = max_features
        self.C = C
        self.random_state = random_state

    def _features(self):
        if self.method not in ("tfidf", "embedding", "hybrid"):
            raise ValueError("method must be 'tfidf', 'embedding' or 'hybrid'.")
        if self.method == "embedding":
            return _StaticEmbeddings(self.encoder)
        options = dict(sublinear_tf=True, max_features=self.max_features, dtype=np.float32)
        features = [
            ("word", TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"\w+|[^\w\s]", **options)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **options)),
        ]
        if self.method == "hybrid":
            features.append(("embedding", _StaticEmbeddings(self.encoder)))
        return FeatureUnion(features)

    def fit(self, X, y):
        """Fit on training documents and labels. Accepts generators as well as lists."""
        documents = _documents(X)
        if isinstance(y, (str, bytes, Mapping)) or getattr(y, "ndim", 1) != 1:
            raise TypeError("Expected a collection of labels.")
        labels = list(y)
        check_consistent_length(documents, labels)
        pipeline = Pipeline([
            ("features", self._features()),
            ("classifier", LogisticRegression(C=self.C, solver="saga", max_iter=500,
                                               random_state=self.random_state, tol=1e-3)),
        ])
        # Publish a new fitted pipeline only after every training step succeeds.
        self.pipeline_ = pipeline.fit(documents, labels)
        return self

    @property
    def classes_(self):
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.classes_

    def predict(self, X):
        """Return one label per document in the input order."""
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.predict(_documents(X))

    def predict_proba(self, X):
        """Return probabilities with one row per input document."""
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.predict_proba(_documents(X))
