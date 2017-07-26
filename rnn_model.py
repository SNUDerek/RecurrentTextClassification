# LSTM/CNN for multiclass sequence classification
# based on:
# http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# help from:
# stacked RNNs: https://github.com/fchollet/keras/issues/160
# bi-RNN: https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py

import numpy as np
import csv, codecs
import dataset
from sklearn.metrics import precision_score, accuracy_score
from sklearn.externals import joblib
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from config import model_dir, model_filename, model_filepath, max_epochs, batchsize, \
                   stop_monitor, stop_delta, stop_epochs, \
                   lstm_cells, embedding_size, max_sent_length, \
                   dropout_rate, CNN_layers, DNN_layers, \
                   do_train, do_save, do_load, do_eval, print_test, \
                   eval_sents, train_verbose, test_verbose, save_path, vocab_size, \
                   sents_filename, classes_filename, model_loss


model = None

# read in data
X_train = np.load(save_path+'X_train.npy')
X_test = np.load(save_path+'X_test.npy')

y_train = np.load(save_path+'y_train.npy')
y_test = np.load(save_path+'y_test.npy')

encoder = joblib.load(save_path+'labelencoder.pkl')

# label list
class_set = encoder.classes_
print("classes:", len(class_set))

print("RNN classification test")
# create the model
# (cnn with) LSTM, using dropout
if do_train == True:

    model = Sequential()

    model.add(Embedding(vocab_size, embedding_size, input_length=max_sent_length))

    for layer in range(CNN_layers):
        model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
        model.add(MaxPooling1D(pool_length=2))

    for layer in range(DNN_layers):
        model.add(Bidirectional(LSTM(lstm_cells, return_sequences=True)))
        model.add(Dropout(dropout_rate))

    model.add(Bidirectional(LSTM(lstm_cells)))

    model.add(Dropout(dropout_rate))

    model.add(Dense(len(class_set), activation='relu'))

    model.add(Dense(len(class_set), activation='sigmoid'))

    model.compile(loss=model_loss, optimizer='adam', metrics=['accuracy'])

    # callbacks for saving and early stoppage
    checkpoint = ModelCheckpoint(model_filepath,
                                 monitor=stop_monitor,
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    earlystop = EarlyStopping(monitor=stop_monitor,
                              min_delta=stop_delta,
                              patience=stop_epochs,
                              verbose=0,
                              mode='auto')
    # tboard = TensorBoard(log_dir='./logs',
    #                      histogram_freq=0,
    #                      write_graph=True,
    #                      write_images=False)
    # tboard.set_model(model)
    callbacks_list = [checkpoint, earlystop]

    if train_verbose == 1: print(model.summary())
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              nb_epoch=max_epochs,
              batch_size=batchsize,
              callbacks=callbacks_list,
              verbose=train_verbose)
    if do_save == True:
        model.save(model_dir+model_filename+'.h5')
        # serialize model to JSON
        model_json = model.to_json()
        with open(model_dir+model_filename+'.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_dir+model_filename+'_weights.h5')
        print("saved model to disk")


# load existing model
if do_load == True:
    del model
    model = load_model(model_dir+model_filename+'.h5')
    model.load_weights(model_dir + model_filename + '_weights.h5')
    print("loaded model from disk")


# Final evaluation of the model
if do_eval == True and model:
    scores = model.evaluate(X_test, y_test, verbose=train_verbose)
    print('')
    print("RNN Accuracy: %.2f%%" % (scores[1]*100),'\n')
    # print('confidence score from normalized activations')

    if print_test == True:

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

        test_idx = np.load(save_path+'test_idx.npy')

        predictions = model.predict(X_test, verbose=test_verbose)
        max_sents = eval_sents
        if max_sents > len(predictions):
            max_sents = len(predictions)
        for i in range(max_sents):
            prediction = list(predictions[i])
            confidence = dataset.max_norm(prediction)
            max_idx = prediction.index(max(prediction))
            guess = class_set[max_idx]
            print(encoder.inverse_transform(y_test[i]), "|", guess, '(conf: %.2f%%)' % confidence, corpus_sents[test_idx[i]] )
