"""Classify a sentence with the bundled Wikipedia articles, entirely offline."""

import argparse
from pathlib import Path

from Smarties import TextClassifier, import_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="The French football team won a soccer match in the league.")
    args = parser.parse_args()

    corpus = Path(__file__).with_name("database_file_custom_name.csv")
    articles = import_database(corpus, split=False).drop_duplicates(subset="Content")
    model = TextClassifier().fit(articles["Content"], articles["Class"])
    probabilities = model.predict_proba([args.text])[0]
    for index in probabilities.argsort()[::-1]:
        print(f"{model.classes_[index]:12} {probabilities[index]:6.1%}")


if __name__ == "__main__":
    main()
