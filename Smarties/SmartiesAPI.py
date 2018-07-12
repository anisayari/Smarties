# coding: utf-8

import pandas as pd
import re
import wikipedia
from sklearn import preprocessing
import json
import numpy as np
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
# from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from . import rake
import urllib
import os
import random

mapping_file = 'mapping.json'
lang='en'
lang_dict={
	"arabic" : "ar",
	"danish" : "da",
	"german" : "de",
    "english" : "en",
	"spanish" : "es",
	"french" : "fr",
    "indonesian" : "id",
    "italian" : "it",
	"japanese" : "ja",
	"dutch" : "nl",
	"portuguese" : "pt",
	"brasilian" : "pt-br",
	"romanian" : "ro",
	"russian" : "ru",
}

def rever_dict(my_dict):
    return {v: k for k, v in my_dict.items()}

def text_to_keyword_dataframe(text, class_labelized,
                              number_of_character_per_word_at_least,
                              number_of_words_at_most_per_phrase,
                              number_of_keywords_appears_in_the_text_at_least):
    # print('Keyword Extraction by NLTK, In Progress...')
    language = rever_dict(lang_dict)[lang]
    rake_object = rake.Rake(language, number_of_character_per_word_at_least, number_of_words_at_most_per_phrase, number_of_keywords_appears_in_the_text_at_least)
    keywords = rake_object.run(text)
    if len(keywords) == 0:
        print(
            '{ERROR] Hum no keyword find... Review your settings. Or I cannot understand this document sorry')
        return None,None
    # remove words with \
    keywords = [i for i in keywords if "\\" not in keywords[0]]
    if len(keywords) == 0:
        raise ValueError('[ERROR] Hum no keyword find for all article, please review your seetings')
    df = pd.DataFrame(keywords)
    df.pivot(columns=0, values=1)
    df.columns = ['KeyWord', "Score"]
    df["class_labelized"] = int(class_labelized)
    df.Score = df.Score.round(2)
    # print(df)
    return df, keywords


def get_df_keyword_from_content(df, content_col, class_col,
                                number_of_character_per_word_at_least,
                           number_of_words_at_most_per_phrase,
                           number_of_keywords_appears_in_the_text_at_least):
    df_k = pd.DataFrame()
    for text in df.groupby([class_col])[content_col]:
        #run keyword extraction for all text concatened
        class_labelized = text[0]
        full_text = ''.join(text[1])
        df, keywords = text_to_keyword_dataframe(full_text, class_labelized,
                                                 number_of_character_per_word_at_least,
                                                 number_of_words_at_most_per_phrase,
                                                 number_of_keywords_appears_in_the_text_at_least
                                                 )
        try:
            if df == None:
                continue
        except TypeError:
            pass
        df_k = df_k.append(df)
    # replace value labelized with mapped value from mapping file
    json_file = open(mapping_file)
    json_str = json_file.read()
    mapping_dico = json.loads(json_str)
    inv_map = {v: k for k, v in mapping_dico.items()}
    df_k["class"] = df_k['class_labelized']
    df_k['class'].replace(inv_map,inplace=True)
    df_k.to_csv("keywords_database.csv".format(content_col), sep=";", index=False)
    return df_k


def sampling_class(df, class_col):
    df_tmp = pd.DataFrame()
    len_min = min(df[class_col].value_counts().tolist())
    for c in df[class_col].unique():
        df_tmp = df_tmp.append(df[df[class_col] == c].sample(len_min), ignore_index=True)
    return df_tmp


def split_content(df, content_col, class_col):
    # split each document by sentences
    # TODO need to be defined
    df = pd.concat([pd.Series(row[class_col], row[content_col].split('.'))
                    for _, row in df.iterrows()]).reset_index()
    df.columns = [content_col, class_col]
    return df


def add_entry_to_json(wiki_dico_path, theme, pageid=0, title=''):
    json_file = open(wiki_dico_path)
    json_str = json_file.read()
    wiki_dico = json.loads(json_str)
    if theme not in wiki_dico:
        wiki_dico[theme] = {}
    if pageid:
        title = title.replace(" ", "_")
        wiki_dico[theme].update(({title: int(pageid)}))
    with open(wiki_dico_path, 'w') as fp:
        json.dump(wiki_dico, fp)
    print("[INFO] {0} had been succesfully added to the knowledge base".format(title.replace('_', '')))


def up_wiki_dico(wiki_dico, max_article_links=None):
    print("[INFO] Start sort and sampling...")
    # Remove common content and sampling
    theme_list = list(wiki_dico.keys())
    intersection_list = []
    for i in range(0, len(theme_list) - 1):
        for j in range(i + 1, len(theme_list)):
            intersection_list += list(
                set(list(wiki_dico[theme_list[i]].keys()))
                    .intersection(
                    list(wiki_dico[theme_list[j]].keys())
                )
            )
    #remove intersection + random sampling
    for key, values in wiki_dico.items():
        list_of_key_without_current_key = list(wiki_dico.keys())
        list_of_key_without_current_key.remove(key)
        if max_article_links != None:
            list_to_check = list(set(values)-set(intersection_list))
            if len(list_to_check)> max_article_links:
                key_random = list(random.sample(list_to_check, max_article_links))
            else:
                print('{} not enough article to be sampled'.format(key))
                key_random=list_to_check
            wiki_dico[key] = {k: v for k, v in values.items() if (
                        k not in intersection_list and k in key_random and k not in list_of_key_without_current_key) or (
                                          k == key)}
        else:
            wiki_dico[key] = {k: v for k, v in values.items() if (
                        k not in intersection_list and k not in list_of_key_without_current_key) or (
                                          k == key)}

        #if a title of a class links is not in a list of links of an other class
        #and title is one of the sampling obtained and title

    print("[INFO] Sort and sampling done")
    return wiki_dico


def get_global_page_id_wiki_links_list(links):
    wiki_dico__for_one_theme = {}
    i = 1
    for link in links:
        wiki_dico__for_one_theme[link] = get_page_id(title=link)
        print('[INFO] Getting links in progress... \r {0}/{1}'.format(i, len(links)), end='')
        i += 1
    return wiki_dico__for_one_theme

def set_lang(lg='english'):
    lg = lg.lower()
    if lg not in list(lang_dict.keys()):
        print('Language supported:')
        print(list(lang_dict.keys()))
        raise NotImplemented('{} is not a language supported'.format(lg))
    global lang
    lang = lang_dict[lg]
    wikipedia.set_lang(lang)

def get_page(page):
    return wikipedia.page(page)

def get_page_id(title):
    parameters = {'action': 'query',
                  'prop': 'revisions',
                  'titles': urllib.parse.quote(title.encode("utf8")),
                  'rvprop': 'content',
                  'format': 'json'}
    queryString = "&".join("%s=%s" % (k, v) for k, v in parameters.items())
    queryString = queryString + "&redirects"
    url = "http://%s.wikipedia.org/w/api.php?%s" % (lang, queryString)
    request = urllib.request.urlopen(url)
    encoding = request.headers.get_content_charset()
    jsonData = request.read().decode(encoding)
    data = json.loads(jsonData)
    data = data["query"]
    if next(iter(data["pages"])) == "-1":
        return 0
    return int(next(iter(data["pages"])))


def construct_wiki_dico(wiki_dico_path, title_theme_list, init=False, find_links=False, max_article_links=100):
    #check if wiki dico exist or created
    if not os.path.isfile(wiki_dico_path):
        with open(wiki_dico_path, 'w') as fp:
            json.dump({}, fp)
        print(wiki_dico_path + ' Succesfully created')
    #load it
    json_file = open(wiki_dico_path)
    json_str = json_file.read()
    wiki_dico = json.loads(json_str)

    for title,theme in title_theme_list:
        #create sublist of wikipedia links sampled
        if theme not in list(wiki_dico.keys()):
            wiki_dico[theme] = {}
        pageid = get_page_id(title)
        if pageid != 0:
            # If page has been found
            if pageid not in wiki_dico[theme]:
                try:
                    add_entry_to_json(wiki_dico_path, theme=theme, pageid=pageid, title=title)
                    # print('Title {} added with sucess to the theme {} !'.format(title,theme))
                except wikipedia.exceptions.DisambiguationError as e:
                    print('[ERROR] Look like I found many related topic about that... select one and try again please:')
                    for word in e.options:
                        print(word)
                if find_links:
                    page_primary = wikipedia.page(pageid=pageid)
                    links = page_primary.links
                dico_tmp = {key: 0 for key in links if key not in list(wiki_dico[theme])}
                for key,value in dico_tmp.items():
                    if key in list(dico_tmp.keys()):
                        wiki_dico[theme][key]=dico_tmp[key]
                    else:
                        wiki_dico[theme][key]=value
            else:
                print(
                    '[ERROR] Hum, look like I already know this Wikipedia Article ' + title + ' try to learn me something else please ! \n')

        else:
            suggest = wikipedia.suggest(title)  # else try to reach a suggestion page
            if suggest is not None:
                print('[INFO] Title {} of the theme {} had not been found, try to reach: !'.format(title, theme, suggest))
                construct_wiki_dico(wiki_dico_path, (suggest, theme))
            else:
                print(
                    '[ERROR] I\'m sorry, but I Cannot find any article or suggestions for %s in the theme, please try again'.format(
                        title, theme))
                pass

    wiki_dico = up_wiki_dico(wiki_dico, max_article_links)

    if find_links:
        for title, theme in title_theme_list:
            print('\n[INFO] Getting Wikipedia links for {}'.format(theme))
            wiki_dico[theme] = get_global_page_id_wiki_links_list(wiki_dico[theme])

    if init:
        with open(wiki_dico_path, 'w') as fp:
            json.dump(wiki_dico, fp)
    else:
        for theme, dico in wiki_dico.items():
            for title, pageid in dico.items():
                add_entry_to_json(wiki_dico_path, theme, pageid=pageid, title=title)

    print('\n[INFO] Construct Knowledge base from Wikipedia DONE.')


def clean_wiki_page(content):
    content = re.sub('=.*=', '', content)  # remove = sign from title
    content = content.replace(";", '')  # delete all ; sign to be sure to not be confuse when saveing df as csv
    content = ''.join(content.splitlines())
    # with open("Output.txt", "w", encoding="utf-8") as text_file:
    #    text_file.write(content)
    return content


def default(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError


def construct_database_from_knwoledge_base(wiki_dico_path, database_file_ouput):
    df_database = pd.DataFrame(columns=["Content", "Class", "SubClass"])
    json_file = open(wiki_dico_path)  # load knwoledge base
    json_str = json_file.read()
    wiki_dico = json.loads(json_str)
    for key, value in wiki_dico.items():
        print('----------------{}----------------'.format(str(key)))
        for key2, value2 in value.items():
            print('Getting page \r{}'.format(str(key2)))
            try:
                page = clean_wiki_page(wikipedia.page(pageid=value2).content)
                df_tmp = pd.DataFrame({"Content": page,
                                       "Class": key,
                                       "SubClass": key2}, index=[0])
                df_database = df_database.append(df_tmp, ignore_index=True)
                df_database.reset_index(inplace=True, drop=True)
            except:
                print('[ERROR] PageID do not valid for {0} with pageid: {1} or Disambiguation Error'.format(key2,value2))
            # add new entry in database

    le = preprocessing.LabelEncoder()
    le.fit(df_database.Class)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    with open(mapping_file, 'w') as fp:
        json.dump(le_name_mapping, fp, default=default)
    print(le_name_mapping)
    df_database["class_labelized"] = df_database.Class.map(le_name_mapping)
    df_database.to_csv(database_file_ouput, header=True, index=False, sep=";")


def label_sentences(df, content_columns="Content", w=None):
    labeled_sentences = []
    for index, datapoint in df.iterrows():
        tokenized_words = re.findall(w, datapoint[content_columns].lower())
        labeled_sentences.append(LabeledSentence(words=tokenized_words, tags=['SENT_%s' % index]))
    return labeled_sentences


def train_doc2vec_model(labeled_sentences):
    model = Doc2Vec(alpha=0.025, min_alpha=0.025)
    model.build_vocab(labeled_sentences)
    for epoch in range(10):
        model.train(labeled_sentences, epochs=model.epochs, total_examples=model.corpus_count)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    return model


def vectorize_comments(df, d2v_model, df_init, action='Train'):
    comments = []
    for i in range(0, df.shape[0]):
        if action == 'Test':
            index = df_init.shape[0] + i - 1
        else:
            index = i
        label = 'SENT_%s' % index
        comments.append(d2v_model.docvecs[label])
    df['vectorized_comments'] = comments
    return df


def train_classifier(X, y):
    n_estimators = [50,100,200]
    min_samples_split = [2,5,7]
    min_samples_leaf = [1,2,3]
    parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf,
                  'min_samples_split': min_samples_split}
    clf = GridSearchCV(RFC(verbose=0, n_jobs=-1), cv=None, param_grid=parameters,verbose=0)
    clf.fit(X, y)
    return clf


def import_database(database_file, content_col="Content", class_col="class_labelized", sampling=False, split=True):
    print('[INFO] Import DATABASE...')
    df = pd.read_csv(database_file, header=0, sep=";", encoding="utf-8")
    if split:
        df = split_content(df, content_col, class_col)
    if sampling:
        df = sampling_class(df, class_col)
    df.reset_index(inplace=True)
    print('[INFO] Import DATABASE DONE')
    return df

def sort_keyword_from_database(df, number_of_character_per_word_at_least,
                               number_of_words_at_most_per_phrase,
                               number_of_keywords_appears_in_the_text_at_least,
                               content_col="Content", class_col="class_labelized", stoppath="Stoplist.txt"):
    print('[INFO] Sort Keyword From Database...')
    df_k = get_df_keyword_from_content(df, content_col, class_col,
                                       number_of_character_per_word_at_least,
                           number_of_words_at_most_per_phrase,
                           number_of_keywords_appears_in_the_text_at_least)
    for keyword_list in df_k.groupby(class_col)['KeyWord']:
        class_labelized = keyword_list[0]
        keyword_list = keyword_list[1]
        for index, row in df.iterrows():
            find = False
            if row[class_col] == class_labelized:
                for keyword in keyword_list:
                    if keyword in row[content_col]:
                        find = True
                if not find:
                    df.drop(index, inplace=True)
    df.reset_index(inplace=True)
    print('[INFO] Sort Keyword From Database DONE')
    return df

def model_from_database(df, content_col="Content", class_col="class_labelized"):
    # Sampling database
    # lmtzr = WordNetLemmatizer()
    w = re.compile("\w+", re.I)
    sen = label_sentences(df, content_columns=content_col, w=w)
    model = train_doc2vec_model(sen)
    df = vectorize_comments(df, model, df)
    X_train, X_test, y_train, y_test = train_test_split(df["vectorized_comments"].T.tolist(),
                                                        df[class_col], test_size=0.10, random_state=4)
    classifier = train_classifier(X_train, y_train)
    # print(classifier.predict(X_test))
    print(classifier.score(X_test, y_test))
    return classifier, df


def predict(clf, df_init, sentence, content_col='Content'):
    data = {'Content': [sentence]}
    df = pd.DataFrame(data)
    w = re.compile("\w+", re.I)
    df = df_init.append(df)
    df.reset_index(drop=True,inplace=True)
    sen = label_sentences(df, content_columns=content_col, w=w)
    model = train_doc2vec_model(sen)
    df = vectorize_comments(df, model, df, action='Train')
    X_test = df["vectorized_comments"].T.tolist()
    X = [X_test[-1]]
    # print(clf.predict(X))
    print(clf.predict_proba(X))
    predicted = clf.predict(X)[0]
    with open(mapping_file) as f_in:
        dict_mapping = json.load(f_in)
    for key, value in dict_mapping.items():
        print('[RESULTS] - Repartition')
        print("I think to  {0} %  : {1}%".format(key,round(float(clf.predict_proba(X)[0][value]) * 100, 1)))

    print('--------------------------------------------')
    for key, value in dict_mapping.items():
        if predicted == value:
            print("I think to  {} % that it is about: {}".format(round(float(clf.predict_proba(X)[0][value]) * 100, 1),
                                                                 key))
            break
