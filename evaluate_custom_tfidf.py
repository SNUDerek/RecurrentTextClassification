import numpy
import TFIDF_intent
import dataset, analysis
from random import randint
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

sents_filename = "out_convers_top10_nva.txt"
classes_filename = "out_labels_top10.txt"
save_path = "data/"

vocab_size = 20000
train_percent = 0.85
checkdata = 0  # idiot check for data formatting
data_verbose = 0  # verbose data processing

print("loading data, please wait...")
sents, classes = dataset.get_lists(sents_filename, classes_filename, testing=data_verbose)

print("making train/test sets, please wait...")
shuffled, train_stop = dataset.shuffle_data(len(sents), train_percent)
X_train, y_train, X_test, y_test, train_set, test_set, train_gold, test_gold, class_set = \
    dataset.get_text_test_train(sents, classes, shuffled, train_stop,
                                trainsize=train_percent,
                                max_vocab=vocab_size - 2,
                                testing=data_verbose)

# just to check if data is looking good
if checkdata != 0:
    print("VISUAL DATA CHECK:")
    print("X_train[:1]", X_train[:2])
    print("y_train[:1]", y_train[:2])
    print("X_test[:1] ", X_test[:2])
    print("y_test[:1] ", y_test[:2])
    print("len y_test ", len(y_test))
    print('')

tfidf_model = TFIDF_intent.classify_tfidf(X_train, y_train, X_test, y_test,
                                          test_set, test_gold, class_set, vocab_size)

# sample random intervals from  each X_test
X_data = [] # random subsampling
gold_data = []
for idx in range(len(X_test)):

    # random numbers to generate sample from conversation = "test sentence"
    conversation = X_test[idx] # sentence as words
    temp = []

    if len(conversation) > 30:

        start_idx = randint(10, len(conversation)-11) # get starting index (padded)
        sent_len = randint(5, 10) # get between 4 and 8 keywords to sim sentence
        # todo: try duplicating sentence to max input length
        multiplier =  int(220 / sent_len)
        for i in range(multiplier):
            temp += conversation[start_idx : start_idx + sent_len]
        X_data.append(temp)
        # todo: else just feed it in
        # X_data.append(conversation[start_idx : start_idx + sent_len])

    else:
        X_data.append(conversation)

y_data = y_test

print("evaluating extracted examples...")
# print('confidence score from normalized activations')
predictions = tfidf_model.predict(X_data)

for i in range(len(predictions)):
    prediction = predictions[i]
    guess = class_set[prediction]
    gold = class_set[y_data[i]]

    print(gold, "|", guess, ' '.join(X_data[i]))

print(classification_report(y_test, predictions))
print("TF-IDF accuracy: %.2f%%" % (accuracy_score(y_test, predictions)*100))
