import Smarties as sm


wiki_dico_path = "wiki_dico.json"
#TODO import and nltk and create function to adapat to each language
#database_file = "corpus.csv"

if __name__ == '__main__':
    sm.ConstructDatabaseFromKnwoledgebase(wiki_dico_path,database_file_ouput="database_file_custom_name.csv")
    df = sm.ImportDatabase(database_file = "database_file_custom_name.csv")

    classifier,model,df = sm.ModelFromDatabase(df)

    sentence_to_predict = "An individual's, nation's, or organization's carbon footprint can be measured by undertaking a GHG emissions assessment or other calculative activities denoted as carbon accounting. "

    sm.Predict(classifier,model,df,sentence_to_predict)
