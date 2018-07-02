import Smarties as tc


wiki_dico_path = "wiki_dico.json"
#TODO import and nltk and create function to adapat to each language
#database_file = "corpus.csv"

if __name__ == '__main__':
    tc.ConstructDatabaseFromKnwoledgebase(wiki_dico_path)
    df = tc.ImportDatabase()

    classifier,model,df = tc.ModelFromDatabase(df)

    sentence_to_predict = "An individual's, nation's, or organization's carbon footprint can be measured by undertaking a GHG emissions assessment or other calculative activities denoted as carbon accounting. "

    tc.Predict(classifier,model,df,sentence_to_predict)
