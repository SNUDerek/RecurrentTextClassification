import codecs, re, random
from collections import Counter
import numpy as np
from config import vocab_size as VOCAB_SIZE


# function to get list of strings from data
# takes corpus as filename
# returns list of sentence token lists
def get_lists(file_corpus, testing=0):
    if testing == 1:
        print('starting dataset.get_lists()...')
    f_corpus = codecs.open(file_corpus, 'rb', encoding='utf8')
    sents = []

    for line in f_corpus:
        sents.append(line.strip('\n'))

    return(sents)


# function to get vocab and inverse vocab
# adds UNK and PAD
# takes list : sents, max vocab #, stoplist
def get_vocab(sents, maxvocab, stoplist=[], testing=0):
    # get vocab list
    vocab = []
    for sent in sents:
        sentlst = sent.split(' ')
        for word in sentlst:
            vocab.append(word)

    counts = Counter(vocab) # get counts of each word
    vocab_set = list(set(vocab)) # get unique vocab list
    sorted_vocab = sorted(vocab_set, key=lambda x: counts[x], reverse=True) # sort by counts
    sorted_vocab = [i for i in sorted_vocab if i not in stoplist]

    if testing == 1:
        print("\ntotal vocab size:", len(sorted_vocab), '\n')
        print(sorted_vocab, '\n')

    sorted_vocab = sorted_vocab[:maxvocab-2]
    vocab_dict = {k: v+1 for v, k in enumerate(sorted_vocab)}
    vocab_dict['UNK'] = maxvocab-1
    vocab_dict['PAD'] = 0
    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}
    return vocab_dict, inv_vocab_dict


# function to convert sents to vectors
# takes list : sents, list : vocab
# returns list of vectors (as lists)
def index_sents(sents, vocab, testing=0):

    if testing==1:
        print("starting vectorize_sents()...")
    vectors = []

    # iterate thru sents
    for sent in sents:
        sent_vect = []
        sentlist = sent.split(' ')
        for word in sentlist:
            if word in vocab.keys():
                idx = vocab[word]
                sent_vect.append(idx)
            else: # out of max_vocab range or OOV
                sent_vect.append(vocab['UNK'])
        vectors.append(sent_vect)
    return(vectors)


# function to split data based on indices
def get_sublist(lst, indices):
    result = []
    for idx in indices:
        result.append(lst[idx])
    return result


# one-hot vectorize function
# takes list of integer indices, max number of labels
# https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
def onehot_vectorize(lst, num):

    return np.eye(num)[lst]


# function to return lexicalized morphs from mecab
# takes sentence as string
# returns space-separated string of lexicalized morphemes
def kkma_tokenize(sents):
    from konlpy.tag import Kkma
    kkma = Kkma()
    lex_sents = []
    # POS-tag and get lexical form from morphemes using KONLPY
    for sent in sents:
        lex_sents.append(' '.join(kkma.morphs(sent)))
        if len(lex_sents) % 200 == 0:
            print("kkma: done", len(lex_sents), "of", len(sents), "total")
    return lex_sents


# function to decode integer-indexed sequence to tokens
# takes integer-indexed sentence, inverse vocab (int to word)
def decode_seq(sent, vocab):
    str = []
    for intr in sent:
        # print(intr)
        str.append(vocab[int(intr)])
    return(str)


# datagenerator so no memory issues
# takes batch size, filepaths to npy-saved data, max vocab, epoch size, class number
# https://github.com/fchollet/keras/issues/2708
# https://github.com/fchollet/keras/issues/1627
def dataGenerator(batch_size,
                  input_filepath='savedata/',
                  xfile='X_train.npy',
                  yfile='y_train.npy',
                  vocabsize=VOCAB_SIZE,
                  epochsize=300000,
                  class_num=10):

    i = 0
    X = np.load(input_filepath + xfile)
    y = np.load(input_filepath + yfile)

    while True:
        # add in data reading/augmenting code here
        y_batch = onehot_vectorize(y[i:i + batch_size], class_num)
        yield (X[i:i + batch_size], y_batch)
        if i + batch_size >= epochsize:
            i = 0
        else:
            i += batch_size


# normalize a list so total prob = ~1
def normalize(list):
    norm = [float(i) / sum(list) for i in list]
    return(norm)

# get maximum normed value from list
# (for """confidence""" value)
def max_norm(list):
    norm = normalize(list)
    return(max(norm))

