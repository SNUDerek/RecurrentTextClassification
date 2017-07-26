import codecs, re
import numpy as np
from dataset import index_sents, get_vocab, get_sublist, onehot_vectorize
import csv
from keras.preprocessing import sequence
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import vocab_size as VOCAB_SIZE
from config import max_sent_length as MAX_SEQ_LENGTH
from config import stoplist, sents_filename, classes_filename
from config import train_percent, save_path


print('Reading in the files...\n')

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

# indices
indices = []
for idx in range(len(corpus_sents)):
    indices.append(idx)


print('Train-test Splitting...\n')
# train-test split with indices
train_idx, test_idx, y_train, y_test = train_test_split(indices, labels, train_size=train_percent)

X_train = get_sublist(corpus_sents, train_idx)
X_test = get_sublist(corpus_sents, test_idx)


print('Creating vocabulary...\n')
# create UNIFIED vocabulary, pos dictionaries
X_vocab, inv_X_vocab = get_vocab(X_train, VOCAB_SIZE, stoplist)


print('Integer-index the inputs and outputs...\n')
# inputs
X_train = index_sents(X_train, X_vocab, VOCAB_SIZE)
X_test = index_sents(X_test, X_vocab, VOCAB_SIZE)
# labels
encoder = LabelEncoder()
encoder.fit(labels)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

class_num = len(encoder.classes_)
y_train = onehot_vectorize(y_train, class_num)
y_test = onehot_vectorize(y_test, class_num)

print('Truncating & zero-padding the data...\n')
# pad sequences with zeros at END of sequence (if too short)
# cut off sequences over MAX_SEQ_LENGTH
X_train = sequence.pad_sequences(X_train, truncating='post', padding='post', maxlen=MAX_SEQ_LENGTH)
X_test = sequence.pad_sequences(X_test, truncating='post', padding='post', maxlen=MAX_SEQ_LENGTH)


print("Vocab zero-idx tests:")
print(inv_X_vocab[0])
print(inv_X_vocab[1])
print(inv_X_vocab[VOCAB_SIZE-1])
print(X_vocab['PAD'])
print(X_vocab['UNK'])

print('\nSaving data...\n')
# save:
# http://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py

# inverse vocabularies for decoding
# load like this:
# read_dictionary = np.load('my_file.npy').item()
# https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping

# word-to-index vocabulary
np.save(save_path+'X_g2i_vocab.npy', X_vocab)
# index-to-word vocabulary
np.save(save_path+'X_i2g_vocab.npy', inv_X_vocab)

# indices of train and test (to get sentences from file)
np.save(save_path+'train_idx.npy', train_idx)
np.save(save_path+'test_idx.npy', test_idx)

# integer-indexed inputs
np.save(save_path+'X_train.npy', X_train)
np.save(save_path+'X_test.npy', X_test)

# integer-indexed lexical forms
np.save(save_path+'y_train.npy', y_train)
np.save(save_path+'y_test.npy', y_test)

# label encoder
# load with xxx = joblib.load('filename.pkl')
joblib.dump(encoder, save_path+'labelencoder.pkl')