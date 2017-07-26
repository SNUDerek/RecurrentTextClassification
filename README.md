# neural_classifier

## CNN-RNN multiclass classification network
dsh9470@snu.ac.kr


## REQUIREMENTS:
'numpy'
`keras` 2.x
`sklearn`
`mlxtend` for one-hot encoding (can change to numpy, sklearn, keras, ...)
`h5py` for saving in hdf5 format
`nltk` for sample corpus generation, only


## TO-DO:
currently displaying results of trained model does not work


## HOW TO RUN:
1. place your sentences and classes text files in the /datasets directory (use `brown_corp_generator.py` to generate toy data with NLTK)
2. edit parameters in config.py (especially CNN and DNN layers)
3. run `rnn_model.py`, compare to `tfidf_model.py`
4. profit! ...maybe


## SAMPLE:
```
TF-IDF baseline test
TF-IDF precision : 0.551792873159
TF-IDF recall    : 0.501596800651
TF-IDF accuracy  : 63.97%

RNN Accuracy: 38.82%
```
... oh well...


## FILES:
`brown_corp_generator.py` : generate sample multiclass classification data using NLTK

`config.py` : configuration file, edit before using

`dataset.py` : vectorizes sentences by frequency, creates one-hot classification vectors

`evaluate.py`: run to evaluate trained (saved) model on all data (OLD!)

`preprocessor.py` : integer-indexes and saves data for neural model

`rnn_model.py` : CNN-RNN classifier, edit parameters before running

`tfidf_model.py` : baseline comparison of TF-IDF classification


## NOTES
if `onehot_vectorize` gives out of memory error, change `preprocessor` to save integer-indexed version (remove the `onehot_vectorize()` lines) and use `traingenerator` to one-hot encode by batch:

see:
`dataGenerator` class in `dataset.py`
https://github.com/fchollet/keras/issues/2708
https://github.com/fchollet/keras/issues/1627


## DEFAULT DIRECTORIES:
/bak : temporary sentence vectors as np arrays (in case of failure)

/datasets : backups for alternate training data (just for storage)

/model : final trained model

/model_data : final vectors for training/test

/logs : tensorflow tensorboard information (not working??)

/temp_models : models saved at each epoch


## NOTES & NOTICES:
- RNN modified from Jason Brownlee : "Sequence Classification with LSTM RNN in Python with Keras"
  http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
- TF-IDF modified from Herbert : "Notebook of TFIDF Logistic regression on IMDB sentiment dataset"
  https://gist.github.com/prinsherbert/92313f15fc814d6eed1e36ab4df1f92d
- sentences vectorized according to word frequency (1 ~ max vocab), 0 for UNK/OOV tokens
- classes as one-hot vectors, can accomodate as many as listed in class text file
- loss function: https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/






