# TFIDF for (multiclass) classification
# based on:
# https://gist.github.com/prinsherbert/92313f15fc814d6eed1e36ab4df1f92d
# changed classification_report(), accuracy_score() with proper args

import numpy, nltk
import dataset, analysis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# fix random seed for reproducibility
seed = 1337
numpy.random.seed(seed)

def classify_tfidf(X_train, y_train, X_test, y_test, test_set, test_gold, class_set, vocab_size):

    # **********************************************
    # CHANGE VARIABLES HERE!!!!!
    # todo: add startup params instead of this mess
    # **********************************************
    # debugging stuff
    checkdata = 0 # idiot check for data formatting

    # END PARAMETERS
    # **********************************************

    print("TF-IDF baseline test")

    # just to check if data is looking good
    if checkdata != 0:
        print("VISUAL DATA CHECK:")
        print("X_train[:1]", X_train[:2])
        print("y_train[:1]", y_train[:2])
        print("X_test[:1] ", X_test[:2])
        print("y_test[:1] ", y_test[:2])
        print("len y_test ", len(y_test))
        print('')

    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
        ('log', LogisticRegression())
    ])

    model.fit(X_train, y_train)

    # Final evaluation of the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("TF-IDF accuracy: %.2f%%" % (accuracy_score(y_test, y_pred)*100))

    return(model)