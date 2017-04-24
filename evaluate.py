# output evaluation of all data
# http://stackoverflow.com/questions/35074549/how-to-load-a-model-from-an-hdf5-file-in-keras

import csv, codecs
import numpy as np
import pandas as pd
import dataset, analysis
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import save_model, load_model, model_from_json
from config import model_filename, save_path, max_sent_length, \
                   eval_filename, do_verbose

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

# combine data for all cases
X_data = X_train + X_test
# need to do this to pad/truncate and reshape array!!
X_data_padded = sequence.pad_sequences(X_data, maxlen=max_sent_length)
y_data = y_train + y_test
gold_data = train_gold + test_gold

# load json and create model
model = load_model(model_filename)
print("loaded model from disk")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("evaluating all examples...")
# print('confidence score from normalized activations')
predictions = model.predict(X_data_padded, verbose=do_verbose)

out_gold = []
out_guess = []
out_conf = []
out_sent = []

for i in range(len(predictions)):
    prediction = list(predictions[i])
    confidence = analysis.max_norm(prediction)
    max_idx = prediction.index(max(prediction))
    guess = class_set[max_idx]
    gold = class_set[y_data[i].index(max(y_data[i]))]

    out_gold.append(gold)
    out_guess.append(guess)
    out_conf.append(confidence)
    out_sent.append(' '.join(gold_data))

    if do_verbose == 1 and i % 100 == 0:
        print(gold, "|", guess, '(conf: %.2f%%)' % (confidence*100),
              ' '.join(gold_data))

# write to dataframe
data = {'gold': out_gold,
        'guess': out_guess,
        'conf': out_conf,
        'sent': out_sent}

out_data = pd.DataFrame(data, columns = ['gold','guess','conf','sent'])
out_data.to_csv(eval_filename, sep='\t', encoding='utf-8')
