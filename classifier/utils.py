import os.path
import os
import pickle
import string

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

stemmer = SnowballStemmer("english")


def preprocess_review(review):
    # remove punctuation
    exclude = set(string.punctuation)
    review = ''.join(ch for ch in review if ch not in exclude)
    # remove stop words
    filtered_words = [word for word in review.split(' ') if word not in stopwords.words('english')]
    # apply stemming
    return ' '.join([stemmer.stem(word) for word in filtered_words])


def decode(text):
    try:
        return str.decode(text, "utf-8", "ignore")
    except Exception as e:
        return ""


def preprocess_review_data(filepath="./data/golden_set.csv", review_field="review", text_prep=None, shuffle=True):
    data = pd.read_csv(filepath)
    # shuffle data
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    data["prep_" + review_field] = data[review_field].apply(lambda review: decode(review))
    data["prep_" + review_field] = data["prep_" + review_field].apply(lambda review: preprocess_review(review))
    if text_prep:
        return data, text_prep.transform(data["prep_" + review_field]), text_prep

    text_prep = Pipeline([("vect", CountVectorizer(min_df=5, ngram_range=(1, 3), stop_words="english")),
                         ("tfidf", TfidfTransformer(norm=None))])
    text_prep.fit(data["prep_" + review_field])
    return data, text_prep.transform(data["prep_" + review_field]), text_prep


def load_or_evaluate_classification(filepath, review_field, categories, cached, k):
    if not cached or not os.path.isfile(os.path.join(".", "internal_data", "results.pkl")):
        results = evaluate_classification(filepath, review_field, categories, k)
    else:
        pkl_file = open(os.path.join(".", "internal_data", "results.pkl"), 'rb')
        results = pickle.load(pkl_file)
    print(results)


def evaluate_classification(filepath, review_field, categories, k):
    all_results = {}
    for cat in categories:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("For category: %s" % cat)
        data, X, _ = preprocess_review_data(filepath=filepath, review_field=review_field)
        y = data[cat]
        splitter = StratifiedShuffleSplit(n_splits=k, test_size=0.2, random_state=0)

        results = {
            "f1_score": [],
            "precision": [],
            "recall": []
        }

        clf = ensemble.GradientBoostingClassifier(n_estimators=500)

        for train_idx, test_idx in splitter.split(X, y):
            print("Split: %s" % train_idx)
            X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y.iloc[train_idx], y.iloc[test_idx]
            clf.fit(X_train.toarray(), y_train)
            y_pred = clf.predict(X_test.toarray())
            not_correct = y_pred != y_test
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            #for index, item in not_correct.iteritems():
            #    if item:
            #        print("%s: %s" % (data.iloc[index]["id"], data.iloc[index]["review"]))
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            results["f1_score"].append(f1_score(y_test, y_pred, average="macro"))
            results["precision"].append(precision_score(y_test, y_pred, average="macro"))
            results["recall"].append(recall_score(y_test, y_pred, average="macro"))
            print(classification_report(y_test, y_pred))

        for score, values in results.iteritems():
            results[score] = sum(values) / len(values)
        all_results[cat] = results
        pkl_file = open(os.path.join(".", "internal_data", "results.pkl"), 'wb')
        pickle.dump(results, pkl_file, -1)
    return all_results


def train_classifier(clf, X_train, y_train):
    clf.fit(X_train.toarray(), y_train)
    return clf


def train_and_save_model(filepath, review_field, categories):
    data, X, text_prep = preprocess_review_data(filepath=filepath, review_field=review_field)
    joblib.dump(text_prep, os.path.join(".", "internal_data", "text_prep.pkl"))
    for category in categories:
        clf = train_classifier(ensemble.GradientBoostingClassifier(verbose=2, n_estimators=500), X, data[category])
        model_details = {"text_field": review_field, "category": category}
        directory = os.path.join(".", "internal_data", category)
        if not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(clf, os.path.join(directory, "model.pkl"))
        joblib.dump(model_details, os.path.join(directory, "model_details.pkl"))


def classify_and_save_results(filepath, categories):
    print(os.getcwd())
    text_prep = joblib.load(os.path.join(".", "internal_data", "text_prep.pkl"))
    model_details = joblib.load(os.path.join(".", "internal_data", "model_details.pkl"))
    data, X, _ = preprocess_review_data(filepath, model_details["text_field"], text_prep, shuffle=False)
    for category in categories:
        print("Classifying category: %s" % category)
        directory = os.path.join(".", "internal_data", category)
        # clf = joblib.load(os.path.join(directory, "model.pkl"))
        model_path = os.path.join(directory, "model.pkl")
        print(model_path)
        clf = joblib.load(model_path)
        print(data.columns)
        y_pred = clf.predict(X.toarray())
        data["PREDICTED_" + category] = y_pred
        if category in data:
            print("For category: %s" % category)
            print(classification_report(data[category], y_pred))

    data.to_csv(filepath[:-4] + "_1.csv", encoding='utf-8', index=False)
