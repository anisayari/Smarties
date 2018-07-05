import Smarties as sm


wiki_dico_path = "wiki_dico.json"
#TODO import and nltk and create function to adapat to each language
#database_file = "corpus.csv"

if __name__ == '__main__':
    sm.ConstructWikiDico(wiki_dico_path, 'Soccer', 'Soccer')
    sm.ConstructWikiDico(wiki_dico_path, 'Baseball', 'Baseball')
    sm.ConstructWikiDico(wiki_dico_path, 'Golf', 'Golf')
    sm.ConstructWikiDico(wiki_dico_path, 'Basketball', 'Basketball')
    sm.ConstructWikiDico(wiki_dico_path, 'Judo', 'Judo')
    
    sm.ConstructDatabaseFromKnwoledgebase(wiki_dico_path,database_file_ouput="database_file_custom_name.csv")
    
    df = sm.ImportDatabase(database_file = "database_file_custom_name.csv")

    classifier,model,df = sm.ModelFromDatabase(df)

    sentence_to_predict = "The French soccer team is perhaps one of the best team around the world."
    sm.Predict(classifier,model,df,sentence_to_predict)
