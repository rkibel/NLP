import sys
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pandas as pd
from pathlib import Path
import re

# read txt file function
# we assume each sentence in the txt file is separated by a newline and a space
# if it is the test txt file, each sentence is separated by a newline
def read_file(file_path, contains_tag = True):
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()
    res = []
    if contains_tag:
        raw_docs = re.split(r'\n \n', raw_text)
        for doc in raw_docs:
            pairs = []
            for line in doc.split('\n'):
                token, tag = line.split(' ')
                pairs.append((token, tag))
            res.append(pairs)
    else:
        raw_docs = re.split(r'\n\n', raw_text)
        for doc in raw_docs:
            pairs = []
            for line in doc.split('\n'):
                pairs.append((line, ''))
            res.append(pairs)
    return res


# feature template
def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-1:]': word[-1:],
        'word[-2:]': word[-2:],
        'word[-3:]': word[-3:],
        'word[:1]': word[:1],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.nonalpha()': not word.isalpha()
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word[-3:]': word1[-3:]
        })
    else:
        features['BOS'] = True
    if i > 1:
        word1 = sent[i-2][0]
        features.update({
            '-2:word.lower()': word1.lower(),
            '-2:word.istitle()': word1.istitle(),
            '-2:word.isupper()': word1.isupper(),
            '-2:word[-3:]': word1[-3:]
        })
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word[-3:]': word1[-3:]
        })
    else:
        features['EOS'] = True
    if i < len(sent)-2:
        word1 = sent[i+2][0]
        features.update({
            '+2:word.lower()': word1.lower(),
            '+2:word.istitle()': word1.istitle(),
            '+2:word.isupper()': word1.isupper(),
            '+2:word[-3:]': word1[-3:]
        })
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

def read_pandas(fname):
    df = pd.read_csv(fname)
    df = df.set_index("id")
    return df

def main():
    test = read_pandas("test_noans.csv")

    train_sents = read_file('train.txt')
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    validation_sents = read_file('validation.txt')
    X_valid = [sent2features(s) for s in validation_sents]
    y_valid = [sent2labels(s) for s in validation_sents]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=20,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_valid)
    labels = list(crf.classes_)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    crf.fit(X_valid, y_valid)

    # warning: this next chunk of CV code took roughly an hour locally
    crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
    )
    params_space = {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }
    f1_scorer = make_scorer(metrics.flat_f1_score, average = 'macro', labels=labels)
    rs = RandomizedSearchCV(crf, params_space, cv=5, verbose=1, n_jobs=-1, n_iter=100, scoring=f1_scorer)
    rs.fit(X_train, y_train)
    
    print('best params:', rs.best_params_)
    print('best CV score:', rs.best_score_)
    print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
    
    crf = rs.best_estimator_
    y_pred = crf.predict(X_valid)
    print(metrics.flat_classification_report(
        y_valid, y_pred, labels=sorted_labels, digits=3
    ))

    y_pred = crf.predict(X_train)
    print(metrics.flat_classification_report(
        y_train, y_pred, labels=sorted_labels, digits=3
    ))

    test_sents = read_file(test_file, contains_tag = False)
    X_test = [sent2features(s) for s in test_sents]
    y_test = crf.predict(X_test)

    flat_y_test = [item for sublist in y_test for item in sublist]
    for i, item in enumerate(flat_y_test):
        match(item):
            case 'B-company': flat_y_test[i] = 0
            case 'B-facility': flat_y_test[i] = 1
            case 'B-geo-loc': flat_y_test[i] = 2
            case 'B-movie': flat_y_test[i] = 3
            case 'B-musicartist': flat_y_test[i] = 4
            case 'B-other': flat_y_test[i] = 5
            case 'B-person': flat_y_test[i] = 6
            case 'B-product': flat_y_test[i] = 7
            case 'B-sportsteam': flat_y_test[i] = 8
            case 'B-tvshow': flat_y_test[i] = 9
            case 'I-company': flat_y_test[i] = 10
            case 'I-facility': flat_y_test[i] = 11
            case 'I-geo-loc': flat_y_test[i] = 12
            case 'I-movie': flat_y_test[i] = 13
            case 'I-musicartist': flat_y_test[i] = 14
            case 'I-other': flat_y_test[i] = 15
            case 'I-person': flat_y_test[i] = 16
            case 'I-product': flat_y_test[i] = 17
            case 'I-sportsteam': flat_y_test[i] = 18
            case 'I-tvshow': flat_y_test[i] = 19
            case 'O': flat_y_test[i] = 20
            case _: flat_y_test[i] = -1

    test["label"] = flat_y_test
    test_ans = test["label"]
    test_ans.to_csv(prediction_file)
    
if __name__ == "__main__":
    test_file = sys.argv[1]
    prediction_file = sys.argv[2]
    main()