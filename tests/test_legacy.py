"""Exercise the legacy public entry points with real dataframes and offline wiki fixtures."""

import io
from importlib.util import find_spec
import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

import Smarties as sm
from Smarties import SmartiesAPI as api
from Smarties import rake


def test_csv_classification_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "articles.csv"
    data = pd.DataFrame({
        "article": ["Solar orbit. Mars planet!", "Mars in orbit.", "Boat sail.", "Sailing boat!"],
        "label": [42, 42, 7, 7],
        "Class": ["Space", "Space", "Sailing", "Sailing"],
    })
    data.to_csv(csv_path, index=False, sep=";")
    documents = sm.import_database(csv_path, content_col="article", class_col="label")
    assert len(documents) == 5
    assert sm.sampling_class(documents, "label")["label"].value_counts().to_dict() == {7: 2, 42: 2}
    model, training = sm.model_from_database(documents, "article", "label")
    expected = model.predict(["solar orbit"])[0]
    monkeypatch.setattr(type(model), "fit", lambda *a, **kw: pytest.fail("predict retrained"))
    assert sm.predict(model, training, "solar orbit", "article") == expected
    report = capsys.readouterr().out
    assert "Space:" in report and "Sailing:" in report


def test_keyword_filter_handles_unicode_custom_stoplist_and_non_numeric_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stoplist = tmp_path / "stoplist.txt"
    stoplist.write_text("# a comment\net\n", encoding="utf-8")
    frame = pd.DataFrame({
        "text": ["Étoile et étoile.", "mer", "Voile et voile."], "topic": ["A", "A", "B"],
    }, index=[15, 16, 17])
    selected = sm.sort_keyword_from_database(frame, 2, 1, 2, "text", "topic", stoplist)
    assert selected["text"].tolist() == ["Étoile et étoile.", "Voile et voile."]
    keywords = pd.read_csv("keywords_database.csv", sep=";")
    assert set(keywords["KeyWord"]) == {"étoile", "voile"}
    assert set(keywords["class"]) == {"A", "B"}
    assert all(keywords["Score"] > 0)
    assert rake.Rake("english", stop_words=["("]).run("étoile") == [("étoile", 1.0)]
    assert rake.Rake("english").run("the rocket and the planet") == [("rocket", 1.0), ("planet", 1.0)]


def test_language_data_is_local():
    pytest.importorskip("stopwordsiso")
    terms = rake.Rake("french").run("la galaxie et la nébuleuse")
    assert terms == [("galaxie", 1.0), ("nébuleuse", 1.0)]


def test_wikipedia_transport_follows_continuations_and_resolves_aliases(monkeypatch):
    calls = []

    def respond(request, *, timeout):
        url = urlparse(request.full_url)
        assert url.scheme == "https" and url.hostname == "fr.wikipedia.org"
        assert request.get_header("User-agent") and 0 < timeout < 60
        params = parse_qs(url.query)
        calls.append(params)
        prop = params["prop"][0]
        if prop == "info|pageprops":
            content = {"query": {"pages": [{"pageid": 8, "title": "Étoile",
                                            "fullurl": "https://fr.wikipedia.org/wiki/Étoile"}]}}
        elif prop == "extracts":
            content = {"query": {"pages": [{"pageid": 8, "extract": "Luminosité"}]}}
        elif prop == "links":
            more = "plcontinue" not in params
            content = {"query": {"pages": [{"links": [{"title": "Soleil" if more else "Véga"}]}]}}
            if more:
                content["continue"] = {"continue": "||", "plcontinue": "8|0|Soleil"}
        else:
            content = {"query": {
                "normalized": [{"from": "petite_étoile", "to": "Petite étoile"}],
                "redirects": [{"from": "Petite étoile", "to": "Étoile"}],
                "pages": [{"pageid": 8, "title": "Étoile"}, {"missing": True, "title": "X"}],
            }}
        return io.BytesIO(json.dumps(content).encode())

    monkeypatch.setattr(api, "lang", "en")
    sm.set_lang("french")
    monkeypatch.setattr(api, "urlopen", respond)
    assert sm.get_global_page_id_wiki_links_list(["petite_étoile", "X"]) == {"petite_étoile": 8, "X": 0}
    page = sm.get_page("Étoile")
    assert (page.title, page.pageid, page.content, page.summary) == ("Étoile", 8, "Luminosité", "Luminosité")
    assert page.links == ["Soleil", "Véga"]
    assert len(calls) == 6
    with pytest.raises(ValueError, match="Unsupported"):
        sm.set_lang("invalid")


def test_build_corpus_incrementally_and_do_not_replace_on_network_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wiki = tmp_path / "wiki.json"
    ids = {"Étoile A": 8, "Boat": 12}
    monkeypatch.setattr(api, "get_page_id", ids.get)
    sm.construct_wiki_dico(wiki, [("Étoile A", "Astres")], init=True, max_article_links=0)
    sm.construct_wiki_dico(wiki, [("Boat", "Boats")], max_article_links=0)
    assert json.loads(wiki.read_text(encoding="utf-8")) == {"Astres": {"Étoile A": 8}, "Boats": {"Boat": 12}}
    def lookup(*, pageid):
        body = {8: "== Titre ==\nÉtoile; brillante.", 12: "Boat\nsails.", 16: "Aardvark dens."}
        return SimpleNamespace(pageid=pageid, content=body[pageid])

    monkeypatch.setattr(api, "get_page", lookup)
    corpus = tmp_path / "corpus.csv"
    sm.construct_database_from_knowledge_base(wiki, corpus)
    documents = sm.import_database(corpus, split=False)
    assert documents["Content"].tolist() == ["Étoile; brillante.", "Boat sails."]
    assert json.loads((tmp_path / "mapping.json").read_text()) == {"Astres": 0, "Boats": 1}
    # New classes sort before the old ones but must not silently renumber existing articles.
    api.add_entry_to_json(wiki, "Aardvarks", pageid=16, title="Another article")
    sm.construct_database_from_knowledge_base(wiki, corpus)
    assert json.loads((tmp_path / "mapping.json").read_text()) == {"Aardvarks": 2, "Astres": 0, "Boats": 1}

    previous = corpus.read_bytes()
    def fail(**kwargs):
        raise TimeoutError("test outage")

    monkeypatch.setattr(api, "get_page", fail)
    with pytest.raises(TimeoutError):
        sm.construct_database_from_knwoledge_base(wiki, corpus)
    assert corpus.read_bytes() == previous


def test_doc2vec_helpers_use_indexed_training_vectors_and_infer_unseen_text():
    pytest.importorskip("gensim")
    frame = pd.DataFrame({"Content": ["tennis racket", "tennis serve", "sail boat", "boat mast"]},
                         index=[8, 12, 43, 51])
    tagged = sm.label_sentences(frame)
    model = sm.train_doc2vec_model(tagged)
    train = sm.vectorize_comments(frame, model, frame)
    assert train.index.tolist() == frame.index.tolist()
    test = pd.DataFrame({"Content": ["tennis tournament"]}, index=[100])
    unseen = sm.vectorize_comments(test, model, frame, action="Test")
    assert unseen["vectorized_comments"].iloc[0].shape == (model.vector_size,)
    classifier = sm.train_classifier(train["vectorized_comments"].tolist(), [1, 1, 2, 2])
    assert classifier.predict(unseen["vectorized_comments"].tolist())[0] in (1, 2)


def test_batched_reciprocal_links_and_optional_graph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    page = SimpleNamespace(title="Science/Orbits", pageid=1, links=["Mars", "Oceans"], language="en")
    monkeypatch.setattr(api, "get_page", lambda *a, **kw: page)

    def query(language, **params):
        assert params["titles"] == "Mars|Oceans"
        assert params["pltitles"] == page.title
        return {"query": {"pages": [
            {"title": "Mars", "pageid": 2, "links": [{"title": page.title}]},
            {"title": "Oceans", "pageid": 3},
        ]}}

    monkeypatch.setattr(api, "_wiki_query", query)
    assert sm.get_sorted_linked_page(1) == ["Mars"]
    assert (tmp_path / "Science_Orbits_links.csv").read_text().splitlines() == [
        "name,num", "Mars,ACCEPTED", "Oceans,REJECTED",
    ]
    wiki = {"Science": {"Science/Orbits": 1, "Mars": 2, "Common": 3},
            "Other": {"Other": 4, "Common": 3}}
    filtered = sm.up_wiki_dico(wiki, 1, seed_titles={"Science": ["Science/Orbits"]})
    assert filtered == {"Science": {"Science/Orbits": 1, "Mars": 2}, "Other": {"Other": 4}}
    if find_spec("networkx"):
        assert set(sm.get_graph_links(["Science"]).neighbors("Science")) == {"Mars", "Oceans"}


def test_pageview_aggregation(monkeypatch):
    mwviews = pytest.importorskip("mwviews.api")

    class Client:
        def __init__(self, user_agent, parallelism):
            assert "Smarties" in user_agent and parallelism == 1

        def article_views(self, project, articles):
            assert project == "en.wikipedia" and articles == ["A B", "Mars"]
            return {"day_one": {"A_B": 4, "Mars": None}, "day_two": {"A_B": 2, "Mars": 3}}

    monkeypatch.setattr(mwviews, "PageviewsClient", Client)
    assert list(sm.get_page_views_dict(["A B", "Mars"]).items()) == [("A B", 6), ("Mars", 3)]


def test_merge_caps_and_deduplicates_existing_ids_before_selecting_links(tmp_path, monkeypatch):
    path = tmp_path / "wiki.json"
    original = {
        "space": {"Seed_article": 10, "Alias": 10, "Shared title": 20, "Venus": 30, "Mars": 40},
        "other": {"Other name for shared title": 20, "other": 50},
    }
    path.write_text(json.dumps(original))
    ids = {"Seed article": 10, "Another seed": 10, "Missing seed": 0}
    monkeypatch.setattr(api, "get_page_id", ids.get)
    sm.construct_wiki_dico(path, [("Seed article", "space")], max_article_links=1)
    assert json.loads(path.read_text()) == {"space": {"Seed article": 10, "Mars": 40}, "other": {"other": 50}}
    after_merge = path.read_bytes()
    with pytest.raises(ValueError, match="seed for both"):
        sm.construct_wiki_dico(path, [("Seed article", "space"), ("Another seed", "other")], init=True)
    with pytest.raises(LookupError, match="Missing seed"):
        sm.construct_wiki_dico(path, [("Missing seed", "space")], init=True)
    assert path.read_bytes() == after_merge
    sm.construct_wiki_dico(path, [("Seed article", "space")], init=True)
    assert json.loads(path.read_text()) == {"space": {"Seed article": 10}, "other": {"other": 50}}


def test_export_old_redirects_and_cross_theme_aliases_only_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wiki = tmp_path / "legacy.json"
    corpus = tmp_path / "out.csv"
    wiki.write_text(json.dumps({"Alpha": {"Alpha": 7, "First": 71, "Shared": 15},
                                "Beta": {"Beta": 9, "First again": 7, "Shared elsewhere": 16}}))

    def lookup(*, pageid):
        # Distinct stored IDs can redirect to the same current Wikipedia page.
        current = {71: 7, 16: 15}.get(pageid, pageid)
        return SimpleNamespace(pageid=current, content=f"document {current}")

    monkeypatch.setattr(api, "get_page", lookup)
    sm.construct_database_from_knowledge_base(wiki, corpus)
    result = pd.read_csv(corpus, sep=";")
    assert result[["Class", "SubClass", "Content"]].values.tolist() == [
        ["Alpha", "Alpha", "document 7"], ["Beta", "Beta", "document 9"],
    ]
