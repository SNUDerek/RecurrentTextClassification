import codecs, re, random, csv
from collections import Counter
import numpy as np

# indexes sentences by vocab frequency list
# reserves 0 for UNKs

# march 14 changes:
# added train-test splitter for tf-idf that returns text strings, classes as ints

# march 16 changes:
# shuffle data same for both models

# march 23 changes:
# save vocabulary file to txt for new decode

# USAGE
# first get lists like this:
# sents, classes = dataset.get_lists(sents_filename, classes_filename)
# then run train-test split like this:
# train_X, train_y, test_X, test_y, test_set, class_set = \
#     dataset.get_test_train(sents, classes, trainsize=0.8, max_vocab=50000):

# function to get lists from data
# takes inputs, classes txt filenames
# returns lists of sentence token lists, classes
def get_lists(file_sents, file_classes, testing=0):
    if testing == 1: print('starting dataset.get_lists()...')
    f_sents = codecs.open(file_sents, 'rb', encoding='utf8')
    f_classes = codecs.open(file_classes, 'rb', encoding='utf8')
    sents = []
    classes = []
    for line in f_sents:
        sents.append(line.strip('\n').split(' '))
    if testing == 1: print('sentences read...')
    for line in f_classes:
        classes.append(line.strip('\n'))
    if testing == 1: print('classes read...')
    return(sents, classes)

# function to get vocab, maxvocab
# takes list : sents
def get_vocab(sents, testing=0):
    if testing == 1: print('starting dataset.get_vocab()...')
    # get vocab list
    vocab = []
    for sent in sents:
        for word in sent:
            vocab.append(word)

    counts = Counter(vocab) # get counts of each word
    vocab_set = list(set(vocab)) # get unique vocab list
    sorted_vocab = sorted(vocab_set, key=lambda x: -counts[x]) # sort by counts

    if testing==1:
        print("get_vocab[:10]:", sorted_vocab[:10])

    return(sorted_vocab)

# function to convert sents to (index) vectors
# takes list : sents, int : max vocab
# returns list of vectors (as lists)
def vectorize_sents(sents, max_vocab, testing=0):

    counter = 0
    # get sorted vocab
    vocab = get_vocab(sents, testing)
    vectors = []
    # iterate thru sents
    if testing==1:
        print("starting vectorize_sents()...")
        print(len(sents),"sentences to vectorize...")
    for sent in sents:
        sent_vect = []
        for word in sent:
            idx = vocab.index(word) + 1 # reserve 0 for UNK / OOV
            if idx < max_vocab: # in max_vocab range
                sent_vect.append(idx)
            else: # out of max_vocab range
                sent_vect.append(0)
        vectors.append(sent_vect)
        counter += 1
        if counter < 10:
            print(sent_vect)
        if counter % 10 == 0:
            print("sentences vectorized:", counter)
        if counter % 1000 == 0:
            np_vector = np.array(vectors)
            np.save('bak/sentvectors_.npy', np_vector)
    if testing==1:
        print("vectorize_sents[:10]:", vectors[:10])
    print("saving final sent vectors...")
    with open("bak/sentvectors.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vectors)
    np_vector = np.array(vectors)
    np.save('bak/sentvectors.npy', np_vector)
    return(vectors, vocab)

# function to convert sents to (index) vectors
# takes list : sents, list : vocab
# returns list of vectors (as lists)
def vectorize_novel_sents(sents, vocab, testing=0):

    counter = 0
    vectors = []
    # iterate thru sents
    if testing==1:
        print("starting vectorize_sents()...")
        print(len(sents),"sentences to vectorize...")
    for sent in sents:
        sent_vect = []
        for word in sent:
            if word in vocab: # in max_vocab range
                idx = vocab.index(word) + 1 # reserve 0 for UNK / OOV
                sent_vect.append(idx)
            else: # out of max_vocab range
                sent_vect.append(0)
        vectors.append(sent_vect)
        counter += 1
        if counter < 10:
            print(sent_vect)
        if counter % 10 == 0:
            print("sentences vectorized:", counter)
    if testing==1:
        print("vectorize_sents[:10]:", vectors[:10])
    print("saving final sent vectors...")

    return(vectors)

# # function to convert classes to one-hot vectors
# takes list : classes
# returns list of vectors (as lists)
def vectorize_classes(classes, testing=0):
    if testing==1:
        print("starting vectorize_classes()...")
    class_set = list(set(classes))
    class_num = len(class_set)
    vectors = []
    # iterate thru classes
    for item in classes:
        class_vector = []
        for z in range(len(class_set)):
            class_vector.append(0)
        idx = class_set.index(item)
        class_vector[idx] = 1
        vectors.append(class_vector)
    if testing==1:
        print("vectorize_classes[:10]:", vectors[:10])
    return(vectors, class_set)

# # function to convert novel input classes to one-hot vectors
# takes list : classes, list : class set
# returns list of vectors (as lists)
def vectorize_novel_classes(classes, class_set, testing=0):
    if testing==1:
        print("starting vectorize_classes()...")
    class_num = len(class_set)
    vectors = []
    # iterate thru classes
    for item in classes:
        class_vector = []
        for z in range(len(class_set)):
            class_vector.append(0)
        if item in class_set:
            idx = class_set.index(item)
            class_vector[idx] = 1
        vectors.append(class_vector)

    return(vectors)

# shuffles data
# do this externally so both can use same train-test
# takes len(sents) and trainsize
# returns shuffled indices as list and stop idx
def shuffle_data(leng, trainsize):
    entries = []
    for i in range(leng):
        entries.append(i)

    # shuffle indices for randomization
    shuffled = random.sample(entries, len(entries))
    # stop size for train set
    train_stop = int(len(shuffled)*trainsize)
    return(shuffled, train_stop)

# function to randomize and test-train split for RNN
# takes sent list, class list, shuffled indices, train stop idx
# returns train sents, train cats, test sents, test cats, test sents, class set
def get_test_train(sents, classes, shuffled, train_stop,
                   trainsize=0.8, max_vocab=50000, testing=0):

    # vocab = get_vocab(sents, testing=testing)
    sent_vectors, vocab =  vectorize_sents(sents, max_vocab, testing=testing)
    class_vectors, class_set = vectorize_classes(classes, testing=testing)

    # get list entry ... list?
    entries = []
    for i in range(len(sent_vectors)):
        entries.append(i)

    train_X = []
    train_y = []
    test_X = []
    test_y = []
    train_set = []
    train_gold = []
    test_set = [] # test set sentences
    test_gold = [] # test set gold classes
    for j in range(len(shuffled)):
        idx = shuffled[j] # get random index
        if j < train_stop:
            train_X.append(sent_vectors[idx])
            train_y.append(class_vectors[idx])
            train_set.append(sents[idx])
            train_gold.append(classes[idx])
        else:
            test_X.append(sent_vectors[idx])
            test_y.append(class_vectors[idx])
            test_set.append(sents[idx])
            test_gold.append(classes[idx])

    return(train_X, train_y, test_X, test_y, train_set, test_set,
           train_gold, test_gold, class_set, vocab)

# function to randomize and test-train split for TFIDF
# takes sent list, class list, shuffled indices, train stop idx
# returns train sents, train cats, test sents, test cats, test sents, class set
def get_text_test_train(sents, classes, shuffled, train_stop,
                        trainsize=0.8, max_vocab=50000, testing=0):

    vocab = get_vocab(sents, testing=testing)
    class_vectors, class_set = vectorize_classes(classes, testing=testing)
    classes_idx = []
    for vector in class_vectors:
        classes_idx.append(vector.index(max(vector)))

    # get list entry ... list?
    entries = []
    for i in range(len(sents)):
        entries.append(i)

    train_X = []
    train_y = []
    test_X = []
    test_y = []
    train_set = []
    train_gold = []
    test_set = [] # test set sentences
    test_gold = [] # test set gold classes
    for j in range(len(shuffled)):
        idx = shuffled[j] # get random index
        if j < train_stop:
            train_X.append(' '.join(sents[idx]))
            train_y.append(classes_idx[idx])
            train_set.append(sents[idx])
            train_gold.append(classes[idx])
        else:
            test_X.append(' '.join(sents[idx]))
            test_y.append(classes_idx[idx])
            test_set.append(sents[idx])
            test_gold.append(classes[idx])

    return(train_X, train_y, test_X, test_y, train_set, test_set,
           train_gold, test_gold, class_set, vocab)

# function to get novel decode
# takes sent list, class list, (trained model's) vocab
# return
def get_novel_decode(sents, vocab, classes, class_set):

    sent_vects = vectorize_novel_sents(sents, vocab)
    class_vects = vectorize_novel_classes(classes, class_set)

    return(sent_vects, class_vects)

def save_csv(list, filename):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list)
    return

def load_csv(filename):
    with codecs.open(filename, "r", encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter=','))
    return(data)

def save_txt(list, filename):
    with open(filename, "w", newline='') as f:
        for item in list:
            f.write(item)
            f.write('\n')
    return

def load_txt(filename):
    data = []
    with codecs.open(filename, "r", encoding='utf-8') as f:
        for item in f:
            data.append(item.strip('\n'))
    return(data)