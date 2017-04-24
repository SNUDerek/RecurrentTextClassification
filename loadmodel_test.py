# test loading model weights from hdf5 file
# http://stackoverflow.com/questions/35074549/how-to-load-a-model-from-an-hdf5-file-in-keras

import numpy, csv, codecs
import dataset, analysis
# import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import save_model, load_model, model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# **********************************************
# CHANGE VARIABLES HERE!!!!!
# **********************************************
# FILENAMES
model_filename = "model_dev_test.h5"
sents_filename = "DATA_sentences_mini.txt"
classes_filename = "DATA_classes_mini.txt"
save_path = "data/"
do_rnn = True
do_tfidf = True

# NETWORK HYPERPARAMS
lstm_cells = 100
embedding_vector_length = 32
max_sent_length = 100
dropout_rate = 0.2
use_CNN = False
use_DNN = True  # deep LSTM network

vocab_size = 16000
train_percent = 0.80
checkdata = 0  # idiot check for data formatting
data_verbose = 0  # verbose data processing
# debugging stuff
train_verbose = 1  # verbose training output
test_verbose = 1  # verbose test/eval output

# END PARAMETERS
# **********************************************

# load data from csv/txt
# http://stackoverflow.com/questions/19838380/building-list-of-lists-from-csv-file
print("loading data from backed up files...")
X_train = dataset.load_csv(save_path + "X_train.csv")
y_train = dataset.load_csv(save_path + "y_train.csv")
X_test = dataset.load_csv(save_path + "X_test.csv")
y_test = dataset.load_csv(save_path + "y_test.csv")
train_set = dataset.load_csv(save_path + "train_set.csv")
train_gold = dataset.load_txt(save_path + "train_gold.txt")
test_set = dataset.load_csv(save_path + "test_set.csv")
test_gold = dataset.load_txt(save_path + "test_gold.txt")
class_set = dataset.load_txt(save_path + "class_set.txt")

# need to do this to pad/truncate and reshape array!!
X_train = sequence.pad_sequences(X_train, maxlen=max_sent_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_sent_length)

# load json and create model
model = load_model(model_filename)
print("loaded model from disk")
# t todo: testing only
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# eval without training
scores = model.evaluate(X_test, y_test, verbose=train_verbose)
print('')
print("RNN Accuracy: %.2f%%" % (scores[1] * 100), '\n')
# print('confidence score from normalized activations')
predictions = model.predict(X_test, verbose=test_verbose)
for i in range(25):
    prediction = list(predictions[i])
    confidence = analysis.max_norm(prediction)
    max_idx = prediction.index(max(prediction))
    guess = class_set[max_idx]
    print(test_gold[i], "|", guess, '(conf: %.2f%%)' % (confidence*10), ' '.join(test_set[i]))
