import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#
# # fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("datasets/processed_bak.csv", header=0)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:26].astype(float)
Y = dataset[:,26]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

print(X[0])
print(encoded_Y[0])
print(encoded_Y)

# model
model = Sequential()

model.add(Dense(26, input_shape=(26,)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(13))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation('sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
print(model.summary())

# train and test
X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.2)

print(X_train[0])
print(y_train[0])
print(X_test[0])
print(y_test[0])

model.fit(X_train, y_train, batch_size=5, nb_epoch=20, verbose=1)

scores = model.evaluate(X_test, y_test, batch_size=5)
print('')
print('Eval model...')
print("Accuracy: %.2f%%" % (scores[1] * 100), '\n')

preds = model.predict(X_test, batch_size=5)
for idx, pred in enumerate(preds[:100]):
    print("real:", y_test[idx], "pred:", pred)



# using http://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

# # baseline model
# def create_model():
#     # create model
#     model = Sequential()
#     model.add(Dense(26, input_shape=(26,)))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(13))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
#     print(model.summary())
#     return model
#
# # evaluate baseline model with standardized dataset
# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(build_fn=create_model, nb_epoch=25, batch_size=5, verbose=1)))
# pipeline = Pipeline(estimators)