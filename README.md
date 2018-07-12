# Smarties :art:


Smarties is a Text Classifier using an innovative approach based on Wikipedia auto-learning to classify 
any documents/text. We use a Machine Learning and Doc2Vec algorithms.

<p align="center">
  <img width="900" height="auto" src="https://media.giphy.com/media/69tPjFPhuRwELvMLpn/giphy.gif">
</p>
Did you see this scene in Avengers: Age of Ultron ? Our learnable programs work kind similar (without the vilain part, we hope... :smile: ).
Our Ai use the most advanced open knowledge base to learn for identifing new categories, and our solution can always learn new related topics. 

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
    sm.set_lang('english')
    #construct the knowledge base to a wiki_dico.json file
    title_theme_list = []
    title_theme_list.append(('Soccer', 'Soccer'))
    title_theme_list.append(('Baseball', 'Baseball'))
    title_theme_list.append(('Golf', 'Golf'))
    title_theme_list.append(('Basketball', 'Basketball'))
    title_theme_list.append(('Judo', 'Judo'))

    sm.construct_wiki_dico(wiki_dico_path, title_theme_list, init=True, max_article_links=30, find_links=True)
    sm.construct_database_from_knwoledge_base(wiki_dico_path, database_file_ouput="database_file_custom_name.csv")

    df = sm.import_database(database_file ="database_file_custom_name.csv", sampling=True)
    df = sm.sort_keyword_from_database(df, stoppath=stoppath, number_of_character_per_word_at_least=5,
                                       number_of_words_at_most_per_phrase=20,
                                       number_of_keywords_appears_in_the_text_at_least=10)

    classifier,df = sm.model_from_database(df)

    sentence_to_predict = "The French soccer team is perhaps one of the best team around the world."
    sm.predict(classifier, df, sentence_to_predict)
```
Language Supported :
- arabic
- danish
- german
- english
- spanish
- french
- indonesian
- italian
- japanese
- dutch
- portuguese
- brasilian
- romanian
- russian

### TODO :grin:
Please feel free to contribute ! Report any bugs in the issue section. 

- [ ] Complete Code documentation and README
- [x] Build a basis code
- [ ] Multi Langage identification and support
- [ ] File format support
- [ ] Text Granularity classification
- [x] Text processing (Clean, tokenizing,...)
- [x] Wikipedia auto-search by class
- [x] Database creation from a json (Wikipedia learning, Datastructure, sampling..)
- [x] Create text classification algorithm (RAKE, GridSearch, RFC, Doc2Vec)
- [x] Input example and return classification
- [ ] Multiple output classification by granularity


### LICENSE

Please be aware that this code is under the GPL3 license. 
You must report each utilisation of this code to the author of this code (anisayari). 
Please push your code using this API on a forked Github repo public. 
