# TFIDF for (multiclass) classification
# based on:
# https://gist.github.com/prinsherbert/92313f15fc814d6eed1e36ab4df1f92d
# changed classification_report(), accuracy_score() with proper args

import codecs, csv
import numpy as np
import nltk
import dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from config import sents_filename, classes_filename, save_path

# fix random seed for reproducibility
seed = 1337
np.random.seed(seed)


# (article body) sentences as list of strings
corpus_sents = []
with codecs.open(sents_filename, 'rU', encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        corpus_sents.append(' '.join(row))

# classes
labels = []
with codecs.open(classes_filename, 'rU', encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        labels.append(' '.join(row))

train_idx = np.load(save_path+'train_idx.npy')
test_idx = np.load(save_path+'test_idx.npy')

X_train = [corpus_sents[i] for i in train_idx]
y_train = [labels[i] for i in train_idx]
X_test = [corpus_sents[i] for i in test_idx]
y_test = [labels[i] for i in test_idx]

encoder = joblib.load(save_path+'labelencoder.pkl')
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

print("TF-IDF baseline test")

model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('log', LogisticRegression())
])

model.fit(X_train, y_train)

# Final evaluation of the model
y_pred = model.predict(X_test)
print("TF-IDF precision :",precision_score(y_test, y_pred, average='macro'))
print("TF-IDF recall    :", recall_score(y_test, y_pred, average='macro'))
print("TF-IDF accuracy  : %.2f%%" % (accuracy_score(y_test, y_pred)*100))
