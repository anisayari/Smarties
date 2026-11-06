"""Compatibility functions for Smarties' original Wikipedia and DataFrame workflow."""

from collections import Counter
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from functools import cached_property
import json
from numbers import Integral
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

from . import rake
from .classifier import TextClassifier

mapping_file = "mapping.json"
lang = "en"
lang_dict = {
    "arabic": "ar", "danish": "da", "german": "de", "english": "en",
    "spanish": "es", "french": "fr", "indonesian": "id", "italian": "it",
    "japanese": "ja", "dutch": "nl", "portuguese": "pt", "brasilian": "pt",
    "romanian": "ro", "russian": "ru",
}

__all__ = [
    "mapping_file", "lang", "lang_dict", "rake",
    "rever_dict", "text_to_keyword_dataframe", "get_df_keyword_from_content",
    "sampling_class", "split_content", "add_entry_to_json", "error_check", "up_wiki_dico",
    "get_global_page_id_wiki_links_list", "set_lang", "get_page", "get_page_id",
    "get_sorted_linked_page", "get_page_views_dict", "get_graph_links", "construct_wiki_dico",
    "clean_wiki_page", "default", "construct_database_from_knwoledge_base",
    "construct_database_from_knowledge_base",
    "label_sentences", "train_doc2vec_model", "vectorize_comments", "train_classifier",
    "import_database", "sort_keyword_from_database", "model_from_database", "predict",
]


def rever_dict(my_dict):
    return {value: key for key, value in my_dict.items()}


@contextmanager
def _atomic_output(path):
    path = Path(path)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            yield output
            output.close()
            os.replace(temporary, path)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


def _read_json(path):
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def _write_json(path, value):
    with _atomic_output(path) as output:
        json.dump(value, output, ensure_ascii=False, indent=2, default=default)
        output.write("\n")


def text_to_keyword_dataframe(text, class_labelized,
                              number_of_character_per_word_at_least,
                              number_of_words_at_most_per_phrase,
                              number_of_keywords_appears_in_the_text_at_least,
                              *, stop_words=None):
    import pandas as pd

    extractor = rake.Rake(lang, number_of_character_per_word_at_least,
                          number_of_words_at_most_per_phrase,
                          number_of_keywords_appears_in_the_text_at_least,
                          stop_words=stop_words)
    keywords = [(word, score) for word, score in extractor.run(text) if "\\" not in word]
    if not keywords:
        return None, None
    result = pd.DataFrame(keywords, columns=["KeyWord", "Score"])
    result["Score"] = result["Score"].round(2)
    result["class_labelized"] = class_labelized
    return result, keywords


def _label_names(df, class_col):
    if "Class" in df and class_col in df:
        return dict(zip(df[class_col], df["Class"]))
    if "smarties_label_names" in df.attrs:
        return df.attrs["smarties_label_names"]
    if Path(mapping_file).is_file():
        return rever_dict(_read_json(mapping_file))
    return {}


def get_df_keyword_from_content(df, content_col, class_col,
                                number_of_character_per_word_at_least,
                                number_of_words_at_most_per_phrase,
                                number_of_keywords_appears_in_the_text_at_least,
                                *, stop_words=None):
    import pandas as pd

    frames = []
    for label, articles in df.groupby(class_col, sort=False)[content_col]:
        result, _ = text_to_keyword_dataframe(
            ". ".join(articles.dropna()), label, number_of_character_per_word_at_least,
            number_of_words_at_most_per_phrase, number_of_keywords_appears_in_the_text_at_least,
            stop_words=stop_words,
        )
        if result is not None:
            frames.append(result)
    if not frames:
        raise ValueError("No keywords found. Try a lower frequency or shorter minimum length.")
    keywords = pd.concat(frames, ignore_index=True)
    names = _label_names(df, class_col)
    keywords["class"] = [names.get(label, label) for label in keywords["class_labelized"]]
    with _atomic_output("keywords_database.csv") as output:
        keywords.to_csv(output, sep=";", index=False)
    return keywords


def sampling_class(df, class_col):
    if df.empty:
        return df.copy()
    groups = df.groupby(class_col, sort=False)
    return groups.sample(n=int(groups.size().min()), random_state=42).reset_index(drop=True)


def split_content(df, content_col, class_col):
    """Split article text for legacy callers. Split train/test by article before calling this."""
    result = df[[content_col, class_col]].copy()
    result[content_col] = result[content_col].str.split(r"[.!?]+")
    result = result.explode(content_col, ignore_index=True)
    result[content_col] = result[content_col].str.strip()
    return result.loc[result[content_col].notna() & result[content_col].ne("")].reset_index(drop=True)


def add_entry_to_json(wiki_dico_path, theme, pageid=0, title=""):
    path = Path(wiki_dico_path)
    knowledge = _read_json(path) if path.exists() else {}
    articles = knowledge.setdefault(theme, {})
    if pageid:
        articles[title.replace(" ", "_")] = int(pageid)
    _write_json(path, knowledge)


def error_check(egg):
    return egg


def _title_key(title):
    normalized = " ".join(title.replace("_", " ").split())
    return normalized[:1].upper() + normalized[1:]


def _article_key(title, pageid):
    numeric_id = int(pageid or 0)
    return numeric_id if numeric_id > 0 else _title_key(title)


def up_wiki_dico(wiki_dico, max_article_links=None, *, seed_titles=None, previous_seeds=None):
    """Keep explicit seeds and up to N unique, single-theme related articles per theme."""
    if max_article_links is not None and (
        not isinstance(max_article_links, Integral) or max_article_links < 0
    ):
        raise ValueError("max_article_links must be a nonnegative integer")
    seed_titles = seed_titles or {}
    previous_seeds = previous_seeds or {}
    candidates = {}
    occurrences = Counter()
    seed_owner = {}
    for theme, articles in wiki_dico.items():
        explicit = seed_titles.get(theme, ())
        seed_names = {_title_key(theme), *(_title_key(title) for title in explicit)}
        by_id = {}
        for title, pageid in articles.items():
            identity = _article_key(title, pageid)
            # A caller-supplied name wins over old spellings and redirect aliases.
            if title in explicit:
                priority = 0
            elif _title_key(title) in seed_names:
                priority = 1
            elif previous_seeds.get(theme) == title:
                priority = 2
            else:
                priority = 3
            item = (priority, title, pageid)
            if identity not in by_id or item[:2] < by_id[identity][:2]:
                by_id[identity] = item
            if priority < 2:
                owner = seed_owner.setdefault(identity, theme)
                if owner != theme:
                    raise ValueError(f"Wikipedia article {identity!r} is a seed for both {owner!r} and {theme!r}")
        occurrences.update(by_id.keys())
        candidates[theme] = by_id

    for theme, by_id in candidates.items():
        seeds, links = {}, []
        for identity, (priority, title, pageid) in by_id.items():
            if identity in seed_owner:
                if seed_owner[identity] == theme:
                    seeds[title] = pageid
            elif occurrences[identity] == 1:
                if priority == 2:
                    seeds[title] = pageid
                else:
                    links.append((title, pageid))
        wiki_dico[theme] = {**seeds, **dict(sorted(links)[:max_article_links])}
    return wiki_dico


def set_lang(lg="english"):
    """Select a Wikipedia language by historical name or ISO language code."""
    global lang
    name = lg.casefold()
    code = lang_dict.get(name, name)
    if code not in lang_dict.values():
        raise ValueError(f"Unsupported language {lg!r}. Choose one of {', '.join(lang_dict)}.")
    lang = code


def _wiki_query(language, **params):
    query = urlencode({"action": "query", "format": "json", "formatversion": 2,
                       "redirects": 1, "maxlag": 5, **params})
    request = Request(f"https://{language}.wikipedia.org/w/api.php?{query}",
                      headers={"User-Agent": "Smarties (https://github.com/anisayari/Smarties)"})
    with urlopen(request, timeout=20) as response:
        data = json.load(response)
    if "error" in data:
        raise RuntimeError(f"MediaWiki query failed: {data['error']}")
    return data


def _wiki_pages(language, **params):
    continuation = {}
    while True:
        response = _wiki_query(language, **params, **continuation)
        yield from response["query"]["pages"]
        continuation = response.get("continue")
        if not continuation:
            break


def _page_record(language, **params):
    page = next(_wiki_pages(language, **params))
    if "missing" in page or "invalid" in page or "disambiguation" in page.get("pageprops", {}):
        raise LookupError(f"Missing or ambiguous Wikipedia article: {page.get('title')}")
    return page


@dataclass
class _WikipediaPage:
    title: str
    pageid: int
    url: str
    language: str

    @cached_property
    def content(self):
        record = _page_record(self.language, pageids=self.pageid, prop="extracts", explaintext=1)
        return record.get("extract", "")

    @cached_property
    def summary(self):
        record = _page_record(self.language, pageids=self.pageid, prop="extracts", explaintext=1, exintro=1)
        return record.get("extract", "")

    @cached_property
    def links(self):
        pages = _wiki_pages(self.language, pageids=self.pageid, prop="links", plnamespace=0, pllimit="max")
        return [entry["title"] for page in pages for entry in page.get("links", [])]


def get_page(page=None, *, pageid=None):
    if (page is None) == (pageid is None):
        raise ValueError("Supply either a Wikipedia title or a pageid")
    identifier = {"pageids": pageid} if pageid is not None else {"titles": page}
    language = lang
    record = _page_record(language, prop="info|pageprops", ppprop="disambiguation",
                          inprop="url", **identifier)
    return _WikipediaPage(record["title"], record["pageid"], record["fullurl"], language)


def _title_batches(titles):
    # Wikimedia limits page counts, and HTTP proxies can reject very long URLs.
    batch = []
    size = 0
    for title in titles:
        if "|" in title:
            raise ValueError("Wikipedia article titles cannot contain a pipe")
        encoded_size = len(quote_plus(title)) + 3
        if batch and (len(batch) >= 40 or size + encoded_size > 5000):
            yield batch
            batch, size = [], 0
        batch.append(title)
        size += encoded_size
    if batch:
        yield batch


def _resolved_titles(titles, *, language, **params):
    """Fetch up to 40 titles at a time and map normalized names and redirects back to inputs."""
    for batch in _title_batches(titles):
        data = _wiki_query(language, titles="|".join(batch), **params)
        query = data["query"]
        aliases = {item["from"]: item["to"] for key in ("normalized", "redirects")
                   for item in query.get(key, [])}
        pages = {record["title"]: record for record in query["pages"]}
        for title in batch:
            normalized = title
            seen = set()
            while normalized in aliases and normalized not in seen:
                seen.add(normalized)
                normalized = aliases[normalized]
            yield title, pages.get(normalized, {})


def get_global_page_id_wiki_links_list(links):
    pages = _resolved_titles(links, language=lang, prop="pageprops", ppprop="disambiguation")
    return {title: record.get("pageid", 0) if "disambiguation" not in record.get("pageprops", {})
            else 0 for title, record in pages}


def get_page_id(title):
    return get_global_page_id_wiki_links_list([title])[title]


def get_sorted_linked_page(pageid):
    page = get_page(pageid=pageid)
    candidates = _resolved_titles(page.links, language=page.language, prop="links",
                                  pltitles=page.title, pllimit="max", plnamespace=0)
    accepted = []
    # Wiki titles can include slashes or Windows device names. Keep the audit CSV in cwd.
    basename = re.sub(r"[^\w-]+", "_", page.title)[:60]
    with _atomic_output(f"{basename}_links.csv") as output:
        writer = csv.writer(output)
        writer.writerow(["name", "num"])
        for title, record in candidates:
            if record.get("links"):
                accepted.append(title)
                status = "ACCEPTED"
            else:
                status = "IGNORED" if "pageid" not in record else "REJECTED"
            writer.writerow([title, status])
    return accepted


def get_page_views_dict(links):
    from mwviews.api import PageviewsClient

    titles = list(links)
    totals = {title: 0 for title in titles}
    if titles:
        client = PageviewsClient(user_agent="Smarties (https://github.com/anisayari/Smarties)", parallelism=1)
        counts = client.article_views(f"{lang}.wikipedia", titles)
        for day in counts.values():
            for title in titles:
                totals[title] += day.get(title.replace(" ", "_")) or day.get(title) or 0
    return dict(sorted(totals.items(), key=lambda pair: pair[1], reverse=True))


def get_graph_links(links):
    """Build and return a NetworkX graph in memory."""
    import networkx as nx

    graph = nx.Graph()
    for title in links:
        graph.add_node(title)
        graph.add_edges_from((title, linked) for linked in get_page(title).links)
    return graph


def construct_wiki_dico(wiki_dico_path, title_theme_list, init=False, find_links=False, max_article_links=None):
    """Merge or rebuild the specified themes, then clean and cap the resulting knowledge base."""
    path = Path(wiki_dico_path)
    existing = _read_json(path) if path.exists() else {}
    title_theme_list = list(title_theme_list)
    rebuilding = {theme for _, theme in title_theme_list} if init else set()
    knowledge = {theme: dict(articles) for theme, articles in existing.items() if theme not in rebuilding}
    # The old format only stores title -> ID. Our writer puts seeds first. Use the
    # first article as a fallback only when there is no title matching the theme;
    # a shared article must never acquire a class solely from this inference.
    previous_seeds = {
        theme: next(iter(articles)) for theme, articles in knowledge.items()
        if articles and not any(_title_key(title) == _title_key(theme) for title in articles)
    }
    seeds = {}
    for title, theme in title_theme_list:
        articles = knowledge.setdefault(theme, {})
        pageid = get_page_id(title)
        if not pageid:
            raise LookupError(f"Missing or ambiguous Wikipedia seed {title!r} for theme {theme!r}")
        seeds.setdefault(theme, set()).add(title)
        articles[title] = pageid
        if find_links:
            for linked in get_sorted_linked_page(pageid):
                articles.setdefault(linked, 0)
    unresolved = {title for articles in knowledge.values() for title, pageid in articles.items() if not pageid}
    found = get_global_page_id_wiki_links_list(sorted(unresolved)) if unresolved else {}
    for theme, articles in knowledge.items():
        resolved = {title: int(pageid or found[title]) for title, pageid in articles.items()}
        knowledge[theme] = {title: pageid for title, pageid in resolved.items() if pageid > 0}
    knowledge = up_wiki_dico(knowledge, max_article_links, seed_titles=seeds, previous_seeds=previous_seeds)
    _write_json(path, knowledge)


def clean_wiki_page(content):
    return " ".join(re.sub(r"(?m)^\s*=+[^\n]*?=+\s*$", " ", content).split())


def default(o):
    if isinstance(o, Integral):
        return int(o)
    raise TypeError(f"Cannot encode {type(o).__name__} as JSON")


def construct_database_from_knwoledge_base(wiki_dico_path, database_file_ouput):
    """Build the legacy semicolon-separated article CSV and mapping.json."""
    import pandas as pd

    # Old dictionaries may contain underscores, redirect aliases or the same ID in
    # several themes. Canonicalize through MediaWiki before assigning training labels.
    knowledge = {}
    canonical_ids = {}
    pages = {}
    for theme, articles in _read_json(wiki_dico_path).items():
        knowledge[theme] = {}
        for title, pageid in articles.items():
            key = _article_key(title, pageid)
            if key not in canonical_ids:
                try:
                    page = get_page(pageid=key) if isinstance(key, int) else get_page(title)
                except LookupError:
                    canonical_ids[key] = None
                else:
                    canonical_ids[key] = page.pageid
                    pages[page.pageid] = page
            if canonical_ids[key] is not None:
                knowledge[theme][title] = canonical_ids[key]
    records = []
    for theme, articles in up_wiki_dico(knowledge).items():
        for title, pageid in articles.items():
            try:
                text = clean_wiki_page(pages[pageid].content)
            except LookupError:
                continue  # The article may be removed between metadata and text requests.
            if text:
                records.append({"Content": text, "Class": theme, "SubClass": title})
    if not records:
        raise ValueError("No usable Wikipedia articles were found")
    frame = pd.DataFrame(records)
    labels = _read_json(mapping_file) if Path(mapping_file).exists() else {}
    if not isinstance(labels, dict) or any(type(value) is not int for value in labels.values()):
        raise ValueError(f"Expected a mapping of names to integer labels in {mapping_file}")
    if len(set(labels.values())) != len(labels):
        raise ValueError(f"Found duplicate label IDs in {mapping_file}")
    next_label = max(labels.values(), default=-1) + 1
    for name in sorted(set(frame["Class"]) - labels.keys()):
        labels[name] = next_label
        next_label += 1
    frame["class_labelized"] = frame["Class"].map(labels)
    # Existing IDs retain their meanings across runs. A write failure leaves the old
    # corpus readable with either the old or the extended label dictionary.
    with _atomic_output(database_file_ouput) as output:
        frame.to_csv(output, index=False, sep=";")
        _write_json(mapping_file, labels)


construct_database_from_knowledge_base = construct_database_from_knwoledge_base


def label_sentences(df, content_columns="Content", w=None):
    from gensim.models.doc2vec import TaggedDocument

    pattern = w or r"\w+"
    return [TaggedDocument(re.findall(pattern, text.lower()), [f"SENT_{index}"])
            for index, text in df[content_columns].items()]


def train_doc2vec_model(labeled_sentences):
    from gensim.models.doc2vec import Doc2Vec

    return Doc2Vec(documents=list(labeled_sentences), vector_size=100, min_count=1,
                   workers=1, epochs=40, seed=42)


def vectorize_comments(df, d2v_model, df_init, action="Train", *, content_col="Content"):
    result = df.copy()
    if action == "Test":
        result["vectorized_comments"] = [
            d2v_model.infer_vector(re.findall(r"\w+", text.lower())) for text in df[content_col]
        ]
    elif action == "Train":
        result["vectorized_comments"] = [d2v_model.dv[f"SENT_{index}"] for index in df.index]
    else:
        raise ValueError("action must be 'Train' or 'Test'")
    return result


def train_classifier(X, y):
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.model_selection import GridSearchCV

    labels = list(y)
    folds = min(5, min(Counter(labels).values(), default=0))
    if folds < 2:
        raise ValueError("Cross-validation needs at least two samples per class")
    search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1),
                          {"n_estimators": [50, 100], "min_samples_leaf": [1, 2]}, cv=folds)
    return search.fit(X, labels)


def import_database(database_file, content_col="Content", class_col="class_labelized", sampling=False, split=True):
    import pandas as pd

    df = pd.read_csv(database_file, sep=";", encoding="utf-8", keep_default_na=False)
    names = dict(zip(df[class_col], df["Class"])) if "Class" in df else {}
    if split:
        df = split_content(df, content_col, class_col)
    if sampling:
        df = sampling_class(df, class_col)
    df = df.reset_index()
    if names:
        df.attrs["smarties_label_names"] = names
    return df


def sort_keyword_from_database(df, number_of_character_per_word_at_least,
                               number_of_words_at_most_per_phrase,
                               number_of_keywords_appears_in_the_text_at_least,
                               content_col="Content", class_col="class_labelized", stoppath="Stoplist.txt"):
    import pandas as pd

    stop_words = None
    if stoppath and (stoppath != "Stoplist.txt" or Path(stoppath).exists()):
        stop_words = rake.load_stop_words(stoppath)
    keywords = get_df_keyword_from_content(
        df, content_col, class_col, number_of_character_per_word_at_least,
        number_of_words_at_most_per_phrase, number_of_keywords_appears_in_the_text_at_least,
        stop_words=stop_words,
    )
    frame = df.reset_index(drop=True)
    keep = pd.Series(False, index=frame.index)
    for label, words in keywords.groupby("class_labelized")["KeyWord"]:
        texts = frame.loc[frame[class_col] == label, content_col]
        pattern = "|".join(re.escape(word) for word in words)
        keep.loc[texts.index] = texts.str.contains(pattern, case=False, na=False)
    return frame.loc[keep].reset_index(drop=True)


def model_from_database(df, content_col="Content", class_col="class_labelized"):
    """Fit the text pipeline once and preserve the historical (classifier, dataframe) result."""
    model = TextClassifier().fit(df[content_col].tolist(), df[class_col].tolist())
    result = df.copy()
    result.attrs["smarties_label_names"] = _label_names(df, class_col)
    return model, result


def predict(clf, df_init, sentence, content_col="Content"):
    """Print per-class probabilities and return the predicted label for a single document."""
    predicted = clf.predict([sentence])[0]
    probabilities = clf.predict_proba([sentence])[0]
    names = df_init.attrs.get("smarties_label_names")
    if names is None:
        names = _label_names(df_init, "class_labelized")
    for label, probability in zip(clf.classes_, probabilities):
        print(f"{names.get(label, label)}: {probability:.1%}")
    print(f"Prediction: {names.get(predicted, predicted)}")
    return predicted
