"""Exercise the installed public API with and without the optional encoder."""

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score

from Smarties import TextClassifier


TEXTS = ["étoile lune", "fusée lune", "orbite lune",
         "football but", "équipe but", "ballon but"]
LABELS = ["espace"] * 3 + ["sport"] * 3


def test_batch_single_and_scores(tmp_path):
    model = TextClassifier().fit(iter(TEXTS), iter(LABELS))
    queries = ["lune", "but"]
    predictions = model.predict(queries)
    scores = model.predict_proba(queries)
    assert predictions.tolist() == ["espace", "sport"]
    assert predictions.tolist() == [model.predict([query])[0] for query in queries]
    np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(model.classes_[scores.argmax(axis=1)], predictions)
    artifact = tmp_path / "model.joblib"
    joblib.dump(model, artifact)
    np.testing.assert_allclose(joblib.load(artifact).predict_proba(queries), scores)


def test_sklearn_cross_validation_and_integer_labels():
    model = TextClassifier(max_features=100)
    assert clone(model).get_params() == model.get_params()
    assert min(cross_val_score(model, TEXTS, [10] * 3 + [42] * 3, cv=3)) == 1
    with pytest.raises(NotFittedError):
        model.predict(["lune"])


@pytest.mark.parametrize("documents", [
    "lune", [], [None], ["  "], [b"bytes"], {"lune": "sport"}, np.array([["lune"]]),
])
def test_invalid_documents(documents):
    with pytest.raises((TypeError, ValueError)):
        TextClassifier().fit(documents, LABELS)


def test_failed_refit_preserves_fitted_state():
    model = TextClassifier().fit(TEXTS, LABELS)
    with pytest.raises(ValueError):
        model.fit(["one", "two"], [0])
    with pytest.raises(ValueError):
        model.fit(["one", "two"], [0, 0])
    assert model.predict(["lune"]).tolist() == ["espace"]


def test_method_and_encoder_validation(tmp_path):
    with pytest.raises(ValueError, match="method"):
        TextClassifier(method="unknown").fit(TEXTS, LABELS)
    with pytest.raises(ValueError, match="local directory"):
        TextClassifier(method="hybrid", encoder=tmp_path / "missing").fit(TEXTS, LABELS)


@pytest.mark.parametrize("method", ["embedding", "hybrid"])
def test_semantic_methods_train_and_reload_offline(method, tmp_path, monkeypatch):
    model2vec = pytest.importorskip("model2vec")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokens = ["[UNK]", "lune", "étoile", "fusée", "orbite", "but", "équipe", "ballon", "football"]
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    vectors = np.array([[0, 0]] + [[1, 0]] * 4 + [[0, 1]] * 4, dtype=np.float32)
    encoder_path = tmp_path / "encoder"
    model2vec.StaticModel(vectors, tokenizer).save_pretrained(encoder_path)
    model = TextClassifier(method=method, encoder=encoder_path).fit(TEXTS, LABELS)

    def forbid_reload(*args, **kwargs):
        raise AssertionError("predict must reuse the fitted encoder")

    monkeypatch.setattr(model2vec.StaticModel, "from_pretrained", forbid_reload)
    assert model.predict(["orbite", "équipe"]).tolist() == ["espace", "sport"]
    artifact = tmp_path / "semantic.joblib"
    joblib.dump(model, artifact)
    assert joblib.load(artifact).predict(["orbite", "équipe"]).tolist() == ["espace", "sport"]
