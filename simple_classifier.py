import tensorflow as tf
from keras import backend as K
import numpy as np

# Scikit-Learn for preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Keras for Neural Network Classifier
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Activation, TimeDistributed, Dense, Dropout, RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import save_model, load_model

