import rnn_model, tfidf_model
import dataset, analysis
from config import sents_filename, classes_filename, save_path, do_rnn, do_tfidf, \
                   vocab_size, train_percent, data_verbose

# todo: run evals here, not in functions

# load the dataset but only keep the top n words, zero the rest
print("loading data, please wait...")
sents, classes = dataset.get_lists(sents_filename, classes_filename, testing=data_verbose)

if do_rnn == True:

    print("making RNN train/test sets, please wait...")
    shuffled, train_stop = dataset.shuffle_data(len(sents), train_percent)
    X_train, y_train, X_test, y_test, train_set, test_set, \
    train_gold, test_gold, class_set, vocab = \
        dataset.get_test_train(sents, classes, shuffled, train_stop,
                               trainsize=train_percent,
                               max_vocab=vocab_size - 2,
                               testing=data_verbose)

    print("backing up processed data to save_path dir...")
    dataset.save_csv(X_train, save_path+"X_train.csv")
    dataset.save_csv(y_train, save_path+"y_train.csv")
    dataset.save_csv(X_test, save_path+"X_test.csv")
    dataset.save_csv(y_test, save_path+"y_test.csv")
    dataset.save_csv(train_set, save_path+"train_set.csv")
    dataset.save_csv(test_set, save_path + "test_set.csv")
    dataset.save_txt(train_gold, save_path+"train_gold.txt")
    dataset.save_txt(test_gold, save_path+"test_gold.txt")
    dataset.save_txt(class_set, save_path+"class_set.txt")
    dataset.save_txt(vocab, save_path + "vocabulary.txt")

    rnn_model = rnn_model.classify_rnn(X_train, y_train, X_test, y_test,
                                        test_set, test_gold, class_set, vocab_size)

if do_tfidf == True:

    print("making TF-IDF train/test sets, please wait...")
    shuffled, train_stop = dataset.shuffle_data(len(sents), train_percent)
    X_train, y_train, X_test, y_test, train_set, test_set, \
    train_gold, test_gold, class_set, vocab = \
        dataset.get_text_test_train(sents, classes, shuffled, train_stop,
                                    trainsize=train_percent,
                                    max_vocab=vocab_size - 2,
                                    testing=data_verbose)

    tfidf_model = tfidf_model.classify_tfidf(X_train, y_train, X_test, y_test,
                                              test_set, test_gold, class_set, vocab_size)
