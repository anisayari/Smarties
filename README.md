# Smarties :art:


Smarties is a Text Classifier using an innovative approach based on Wikipedia auto-learning to classify 
any documents/text. We use a Machine Learning and Doc2Vec algorithms.

## Getting Started :beginner:

These instructions will get you a copy of the project up and running on your local machine 
for development and testing purposes. See deployment for notes on how to deploy the project 
on a live system.

### Prerequisites

You need to have python 3.+ install on your machine. 

### Installing
```
pip install Smarties
```

### Example :round_pushpin:
```
wiki_dico_path = "wiki_dico.json"

if __name__ == '__main__':
    #construct the knowledge base to a wiki_dico.json file
    sm.ConstructWikiDico(wiki_dico_path, 'Soccer', 'Soccer')
    sm.ConstructWikiDico(wiki_dico_path, 'Baseball', 'Baseball')
    sm.ConstructWikiDico(wiki_dico_path, 'Golf', 'Golf')
    sm.ConstructWikiDico(wiki_dico_path, 'Basketball', 'Basketball')
    sm.ConstructWikiDico(wiki_dico_path, 'Judo', 'Judo')
    
    #Construct and import database from our database previously defined
    sm.ConstructDatabaseFromKnwoledgebase(wiki_dico_path,database_file_ouput="database_file_custom_name.csv")
    df = sm.ImportDatabase(database_file = "database_file_custom_name.csv")

    #Run model training
    classifier,model,df = sm.ModelFromDatabase(df)

    sentence_to_predict = "The French soccer team is perhaps one of the best team around the world."

    #Predict the class of our sample
    sm.Predict(classifier,model,df,sentence_to_predict)
```

### TODO :grin:
Please feel free to contribute ! Report any bugs in the issue section. 

- [ ] Complete Code documentation and README
- [x] Build a basis code
- [x] Text processing (Clean, tokenizing,...)
- [x] Wikipedia auto-search by class
- [x] Database creation from a json (Wikipedia learning, Datastructure, sampling..)
- [x] Create text classification algorithm (RAKE, GridSearch, RFC, Doc2Vec)
- [x] Input example and return classification

### LICENSE

Please be aware that this code is under the GPL3 license. 
You must report each utilisation of this code to the author of this code (anisayari). 
Please push your code using this API on a forked Github repo public. 
