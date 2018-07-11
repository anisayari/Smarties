import Smarties as sm

wiki_dico_path = "wiki_dico.json"
#TODO import and nltk and create function to adapat to each language
stoppath = "Stoplist.txt"

#database_file = "corpus.csv"


if __name__ == '__main__':
    title_theme_list = []
    title_theme_list.append(('Soccer', 'Soccer'))
    title_theme_list.append(('Baseball', 'Baseball'))
    title_theme_list.append(('Golf', 'Golf'))
    title_theme_list.append(('Basketball', 'Basketball'))
    title_theme_list.append(('Judo', 'Judo'))

    sm.ConstructWikiDico(wiki_dico_path, title_theme_list, init=True,max_article_links=30,find_links=True)
    sm.ConstructDatabaseFromKnwoledgebase(wiki_dico_path,database_file_ouput="database_file_custom_name.csv")
    
    df = sm.ImportDatabase(database_file = "database_file_custom_name.csv",sampling=True)
    df = sm.SortKeywordFromDatabase(df,stoppath=stoppath,number_of_character_per_word_at_least=5,
                           number_of_words_at_most_per_phrase=20,
                           number_of_keywords_appears_in_the_text_at_least=10)

    classifier,df = sm.ModelFromDatabase(df)
    
    sentence_to_predict = "The French soccer team is perhaps one of the best team around the world."
    sm.Predict(classifier,df,sentence_to_predict)