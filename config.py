# **********************************************
# EVALUATIONS: RNN and TF-IDF CLASSIFICATION
# **********************************************
do_rnn = True
do_tfidf = True

# **********************************************
# DATA AND SAVE LOCATIONS ETC
# **********************************************
# train-test data paths
sents_filename = "datasets/brown_sents.txt"
classes_filename = "datasets/brown_topics.txt"
# novel decode data paths (for evaluate_custom.py)
eval_sents_filename = "datasets/brown_sents.txt"
eval_classes_filename = "datasets/brown_topics.txt"

data_verbose = 0  # verbose data processing
# directory for data backup (train, test, vocab, etc)
save_path = "model_data/"
# final model savepath & filename
model_filename = "model/model_brown.h5"
# temporary model savepath & filename
model_filepath = "temp_models/weights-{epoch:02d}-{val_acc:.2f}.hdf5"

# **********************************************
# MODEL HYPERPARAMETERS ETC
# **********************************************
# NETWORK TRAINING PARAMS
vocab_size = 18000
train_percent = 0.95
max_epochs = 100         # hard epoch limit; see early stop
batchsize = 32           # items per training batch
stop_monitor = 'val_acc' # variable for early stop: val_loss, val_acc, etc...?
stop_delta = 0.00        # minimum delta before early stop
stop_epochs = 1          # how many epochs to do after stop condition (default = 0)

# NETWORK HYPERPARAMS
lstm_cells = 100        # LSTM cells per layer
embedding_size = 32     # embedding layer size
max_sent_length = 200   # maximum input length (domain: 200+, intent: ~100)
dropout_rate = 0.2      # dropout
CNN_layers = 1          # convolutional layers (try 2 for domain, 0 for intent)
DNN_layers = 2          # deep LSTM network layers (0 = 1-layer model)
# use_GPU = True        # not using at the moment

# ACTIVITY PARAMS
do_train = True         # train model?
do_save = True          # save model (after training)?
do_load = False         # load model (turn off train to load old model)?
do_eval = True          # evaluate model?
print_test = True       # show individual test evaluations
eval_sents = 100          # max number of eval sents to display

# debugging stuff
train_verbose = 1       # verbose training output
test_verbose = 1        # verbose test/eval output

# **********************************************
# EVAL STUFF
# **********************************************
# evaluation file path and filename
eval_filename = "model/speechact_evaluation_data.csv"
do_verbose = 1  # verbose output

# END PARAMETERS
# **********************************************