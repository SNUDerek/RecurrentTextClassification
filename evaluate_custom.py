# output evaluation of all data
# http://stackoverflow.com/questions/35074549/how-to-load-a-model-from-an-hdf5-file-in-keras

import csv, codecs
import numpy as np
import pandas as pd
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
import pandas
from random import randint
from config import model_filename, eval_filename, save_path, \
                   data_verbose, do_verbose, max_sent_length, \
                   eval_sents_filename, eval_classes_filename

checkdata = False  # idiot check for data formatting

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
vocab = dataset.load_txt(save_path + "vocabulary.txt")

sents, classes = dataset.get_lists(eval_sents_filename,
                                   eval_classes_filename, testing=0)
X_data, y_data = dataset.get_novel_decode(sents, vocab, classes, class_set)

# get sentences (the name is misleading, sorry)
gold_data = dataset.load_txt(eval_sents_filename)

# for idx in range(len(X_data)):

    # random numbers to generate sample from conversation = "test sentence"
    # conversation = X_data[idx] # indexed sentence
    # gold_sent = test_set[idx] # sentence as words
    # temp = []

    # sent_len = len(conversation)
    # todo: try duplicating sentence to max input length
    # multiplier =  int(max_sent_length / sent_len)
    # for i in range(multiplier):
    #     temp += conversation
    # X_data_new.append(temp)
    # todo: else just feed it in (works a lot worse)
    # X_data.append(conversation)

# todo: just for testing
if checkdata == True:
    print(max_sent_length)
    print("X_data", X_data[0])
    print("y_data", y_data[0])
    print("gold_data", gold_data[0])

# need to do this to pad/truncate and reshape array!!
X_data_padded = sequence.pad_sequences(X_data, maxlen=max_sent_length)

# load json and create model
model = load_model(model_filename)
print("loaded model from disk")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("evaluating extracted examples...")
# print('confidence score from normalized activations')
predictions = model.predict(X_data_padded, verbose=do_verbose)

out_guess = []
out_conf = []
out_sent = []

for i in range(len(predictions)):
    prediction = list(predictions[i])
    confidence = analysis.max_norm(prediction)
    max_idx = prediction.index(max(prediction))
    guess = class_set[max_idx]
    gold = class_set[y_data[i].index(max(y_data[i]))]

    out_guess.append(guess)
    out_conf.append(confidence)
    out_sent.append(gold_data[i])

    if do_verbose == 1 and i % 100 == 0:
        print(gold, "|", guess, '(conf: %.2f%%)' % (confidence*100),
              gold_data[i])

# write to dataframe
data = {'sent': out_sent,
        'guess': out_guess,
        'conf': out_conf
        }

print("saving", eval_filename, "csv file to dir")
out_data = pd.DataFrame(data, columns = ['guess', 'conf', 'sent'])
out_data.to_csv(eval_filename, sep='\t', encoding='utf-8')
print("...saved!")
# for i in range(len(predictions)):
#     prediction = list(predictions[i])
#     confidence = analysis.max_norm(prediction)
#     max_idx = prediction.index(max(prediction))
#     guess = class_set[max_idx]
#     gold = gold_data[i]
#
#     if do_verbose == 1:
#         print(gold, "|", guess, '(conf: %.2f%%)' % (confidence*100),
#               ' '.join(sents[i]))
#
# scores = model.evaluate(X_data_padded, y_data, verbose=0)
# print('')
# print("RNN Accuracy: %.2f%%" % (scores[1]*100),'\n')
