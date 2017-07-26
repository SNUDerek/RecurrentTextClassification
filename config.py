
# **********************************************
# DATA AND SAVE LOCATIONS ETC
# **********************************************
# train-test data paths
sents_filename = "datasets/brown_sents.csv"
classes_filename = "datasets/brown_topics.csv"

data_verbose = 0  # verbose data processing
# directory for data backup (train, test, vocab, etc)
save_path = "model_data/"
# final model savepath & filename
model_dir = 'model/'
model_filename = "model_brown"
# temporary model savepath & filename
model_filepath = "model_checkpoints/weights-{epoch:02d}-{val_acc:.2f}.hdf5"

# **********************************************
# NLP STUFF: STOPLIST, ETC
# **********************************************
stoplist = []

# **********************************************
# MODEL HYPERPARAMETERS ETC
# **********************************************
# NETWORK TRAINING PARAMS
vocab_size = 18000
train_percent = 0.85
model_loss = 'categorical_crossentropy'  # change according to binary or multiclass
max_epochs = 100         # hard epoch limit; see early stop
batchsize = 32           # items per training batch
stop_monitor = 'val_acc' # variable for early stop: val_loss, val_acc, etc...?
stop_delta = 0.00        # minimum delta before early stop
stop_epochs = 1          # how many epochs to do after stop condition (default = 0)

# NETWORK HYPERPARAMS
lstm_cells = 128        # LSTM features per layer
embedding_size = 64     # embedding layer size
max_sent_length = 128   # maximum input length
dropout_rate = 0.2      # dropout
CNN_layers = 0          # convolutional layers
DNN_layers = 1          # deep LSTM network layers (0 = 1-layer model)
# use_GPU = True        # not using at the moment

# ACTIVITY PARAMS
do_train = True         # train model
do_save = True          # save model
do_load = False         # load model
do_eval = True          # evaluate model
print_test = True       # show individual test evaluations
eval_sents = 100        # max number of eval sents to display

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