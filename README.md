# Smarties


Smarties is a Text Classifier using an innovative approach based on Wikipedia auto-learning to classify 
any documents/text. We use a Machine Learning and Doc2Vec algorithms.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine 
for development and testing purposes. See deployment for notes on how to deploy the project 
on a live system.

### Prerequisites

You need to have python 3.+ install on your machine. 

### Installing
```
pip install Smarties
```

### Example
```
if __name__ == '__main__':
    sm.ConstructDatabaseFromKnwoledgebase(wiki_dico_path)
    df = sm.ImportDatabase()

    classifier,model,df = sm.ModelFromDatabase(df)

    sentence_to_predict = "An individual's, nation's, or organization's carbon footprint can be measured by undertaking a GHG emissions assessment or other calculative activities denoted as carbon accounting. "

    sm.Predict(classifier,model,df,sentence_to_predict)
```

### TODO
Please feel free to contribute ! Report any bugs in the issue section. 

- [ ] Complete Code documentation and README
- [x] Build a basis code
- [x] Text processing (Clean, tokenizing,...)
- [ ] Wikipedia auto-search by class
- [x] Database creation from a json (Wikipedia learning, Datastructure, sampling..)
- [x] Create text classification algorithm (GridSearch, RFC, Doc2Vec)
- [x] Input example and return classification

### LICENSE

Please be aware that this code is under the GPL3 license. 
You must report each utilisation of this code the author of this code. 
Please push your code using this API on a forked Github repo public. 