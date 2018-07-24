# coding: utf-8

import Smarties as sm
import time

wiki_dico_path = "wiki_dico.json"

#database_file = "corpus.csv"


if __name__ == '__main__':
    start_time = time.time()
    sm.set_lang('english')
    title_theme_list = []
    title_theme_list.append(('Soccer', 'Soccer'))
    title_theme_list.append(('Baseball', 'Baseball'))
    title_theme_list.append(('Golf', 'Golf'))
    title_theme_list.append(('Basketball', 'Basketball'))
    title_theme_list.append(('Judo', 'Judo'))

    sm.construct_wiki_dico(wiki_dico_path, title_theme_list, init=True, max_article_links=5, find_links=True)
    sm.construct_database_from_knwoledge_base(wiki_dico_path, database_file_ouput="database_file_custom_name.csv")

    df = sm.import_database(database_file="database_file_custom_name.csv", sampling=True)
    df = sm.sort_keyword_from_database(df, number_of_character_per_word_at_least=2,
                                       number_of_words_at_most_per_phrase=20,
                                       number_of_keywords_appears_in_the_text_at_least=2)

    classifier, df = sm.model_from_database(df)

    sentence_to_predict = "The French soccer team is perhaps one of the best team around the world."
    sm.predict(classifier, df, sentence_to_predict)
    print("--- %s seconds ---" % (time.time() - start_time))
