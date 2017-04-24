# LSTM/CNN for multiclass sequence classification
# based on:
# http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# help from:
# stacked RNNs: https://github.com/fchollet/keras/issues/160
# bi-RNN: https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py
# multi-gpu: https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

# march 10 changes
# BIDRECTIONAL RNNS!
# saving, loading models
# now eval displays gold tag as well
# pass vocab_size-1 to dataset.py b/c 'reserve' 0 for UNKs (changed dataset.py too)
# try GPU support (seems OK?)
# needs multi-GPU support
# still no command line arguments

# march 14 changes
# geez ok now it's a function
# a very, very, very terrible function

# march 16 changes
# model saving, early termination, tensorboard(?)
# get rid of forced CPU for now

# march 19 changes
# small fixes in early stop and model saving, evaluation

import numpy
import dataset, analysis
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from config import model_filename, model_filepath, max_epochs, batchsize, \
                   stop_monitor, stop_delta, stop_epochs, \
                   lstm_cells, embedding_size, max_sent_length, \
                   dropout_rate, CNN_layers, DNN_layers, \
                   do_train, do_save, do_load, do_eval, print_test, \
                   eval_sents, train_verbose, test_verbose

# fix random seed for reproducibility
seed = 1337
numpy.random.seed(seed)

# very terrible RNN function
def classify_rnn(X_train, y_train, X_test, y_test, test_set, test_gold, class_set, vocab_size):

    # truncate and pad input sequences
    X_train = sequence.pad_sequences(X_train, maxlen=max_sent_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sent_length)

    # tensorflow stuff for multi_GPU...?
    sess = tf.Session()
    K.set_session(sess)
    model = 'none' # no model check

    print("RNN classification test")
    # create the model
    # (cnn with) LSTM, using dropout
    # todo: tweak here, fix so not duplicating code
    # todo: GPU support ok??
    if do_train == True:
        # if use_GPU == True:
        # with tf.device('/gpu:0'):
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_size, input_length=max_sent_length))
        for layer in range(CNN_layers):
            model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
            model.add(MaxPooling1D(pool_length=2))
        # todo: add explicit multi-GPU support here?!?!
        for layer in range(DNN_layers):
            model.add(Bidirectional(LSTM(lstm_cells, return_sequences=True)))
            model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(lstm_cells)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(len(class_set), activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

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
                  validation_data=(X_test,y_test),
                  nb_epoch=max_epochs,
                  batch_size=batchsize,
                  callbacks=callbacks_list,
                  verbose=train_verbose)
        if do_save == True:
            model.save(model_filename)
            print("saved model to disk")

    # load existing model
    if do_load == True:
        del model
        model = load_model(model_filename)
        print("loaded model from disk")
        # todo: testing only
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # scores = model.evaluate(X_test, y_test, verbose=0)
        # print('')
        # print("RNN Accuracy: %.2f%%" % (scores[1] * 100), '\n')
        # predictions = model.predict(X_test, verbose=test_verbose)
        # for i in range(eval_sents):
        #     prediction = list(predictions[i])
        #     confidence = analysis.max_norm(prediction)
        #     max_idx = prediction.index(max(prediction))
        #     guess = class_set[max_idx]
        #     print(test_gold[i], "|", guess, '(conf: %.2f%%)' % confidence, ' '.join(test_set[i]))

    # Final evaluation of the model
    if do_eval == True and model != 'none':
        scores = model.evaluate(X_test, y_test, verbose=train_verbose)
        print('')
        print("RNN Accuracy: %.2f%%" % (scores[1]*100),'\n')
        # print('confidence score from normalized activations')

        if print_test == True:
            predictions = model.predict(X_test, verbose=test_verbose)
            max_sents = eval_sents
            if max_sents > len(predictions):
                max_sents = len(predictions)
            for i in range(max_sents):
                prediction = list(predictions[i])
                confidence = analysis.max_norm(prediction)
                max_idx = prediction.index(max(prediction))
                guess = class_set[max_idx]
                print(test_gold[i],"|",guess,'(conf: %.2f%%)' % confidence, ' '.join(test_set[i]) )

    return(model)