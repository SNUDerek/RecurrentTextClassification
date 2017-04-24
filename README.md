# neural_classifier

## CNN-RNN multiclass classification network
dsh9470@snu.ac.kr

## HOW TO RUN:
1. place your sentences and classes text files in the /datasets directory (use `brown_corp_generator.py` to generate toy data with NLTK)
2. edit parameters in config.py
3. run main.py
4. profit! ...maybe

## NOTES:
NB: this is not designed for amazing efficiency, especially the preprocessing. `sklearn` preprocessing would help a lot.

## FILES:
`main.py` : main file, run to use

`config.py` : configuration file, edit before using

`evaluate.py`: run to evaluate trained (saved) model on all data

`rnn_model.py` : CNN-RNN classifier, edit parameters before running

`tfidf_model.py` : baseline comparison of TF-IDF classification

`dataset.py` : vectorizes sentences by frequency, creates one-hot classification vectors

`analysis.py` : (to-do) various tools for evaluation and analysis

`brown_corp_generator.py` : generate sample classification data using NLTK

`loadmodel_test.py` : test code for loading saved models


## DEFAULT DIRECTORIES:
/bak : temporary sentence vectors as np arrays (in case of failure)

/datasets : backups for alternate training data (just for storage)

/model : final trained model

/model_data : final vectors for training/test

/logs : tensorflow tensorboard information (not working??)

/temp_models : models saved at each epoch


## TO-DO LIST:
- swap out crappy vectorization for sklearn vectorizer?
- add command line parameter functionality
- tweak network architecture for better performance
- add more analysis tools



## NOTES & NOTICES:
- RNN modified from Jason Brownlee : "Sequence Classification with LSTM RNN in Python with Keras"
  http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
- TF-IDF modified from Herbert : "Notebook of TFIDF Logistic regression on IMDB sentiment dataset"
  https://gist.github.com/prinsherbert/92313f15fc814d6eed1e36ab4df1f92d
- sentences vectorized according to word frequency (1 ~ max vocab), 0 for UNK/OOV tokens
- classes as one-hot vectors, can accomodate as many as listed in class text file
- loss function: https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/


## CHANGELOG:

March 14:
- added TFIDF classification for baseline comparison
- added function... functionality for RNN, TFIDF (WIP)
- added main.py to run both models to compare

March 16:
- RNN and TFIDF now use same shuffled data for training and testing
- in-training model saving and early stops with keras callbacks
- removed forcing to CPU
- saving and loading tested working:
  vectorized data saved to csv files
  using loaded and saved model
- tensorboard using keras callback (not working)

March 17:
- evaluate.py for full evaluation output to csv
- saving and loading functionality expanded:
- backup training input and gold files, and vocabulary, as well

March 23:
- saving vocabulary *actually* working
- now all variables in config.py






