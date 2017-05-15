# coding: utf-8

from keras.callbacks import TensorBoard
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Lambda, Input, Dropout
from keras.layers import LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers.wrappers import Bidirectional
from keras.datasets import imdb
import keras
import numpy as np
import pickle
import random


print('Opening..')
with open('data/plain_dataset.pickle', 'rb') as fin:
    X, Y = pickle.load(fin)


# Shuffle
NEED_SHUFFLE = False
if NEED_SHUFFLE:
    print('Shuffling..')
    rand_indices = list(range(len(X)))
    random.shuffle(rand_indices)
    rand_X = []
    rand_Y = []
    for i in range(len(X)):
        t = rand_indices[i]
        rand_X.append(X[t])
        rand_Y.append(Y[t])
    X, Y = rand_X, rand_Y


print('Preparing..')
num_alphabet = 36  # (3+33)
num_cat = 99 # (1+98)

SAMPLES      = 1000000  # 1000000
TEST_SAMPLES = 10000
num_train = SAMPLES
num_test = TEST_SAMPLES

train_X = X[:num_train]
train_Y = Y[:num_train]
test_X = X[num_train:(num_train + num_test)]
test_Y = Y[num_train:(num_train + num_test)]

train_Y = keras.utils.to_categorical(train_Y, num_cat)
test_Y = keras.utils.to_categorical(test_Y, num_cat)


train_X = sequence.pad_sequences(train_X, maxlen=MAXLEN)
test_X = sequence.pad_sequences(test_X, maxlen=MAXLEN)
print('x train shape:', train_X.shape, 'y:', train_Y.shape)
print('x test shape:', test_X.shape, 'y:', test_Y.shape)

def binarize(x, sz=num_alphabet):
    from keras.backend import tf
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))
def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 36  # num_alphabet


print('Building model...')

MAXLEN = 120
in_sentence = Input(shape=(MAXLEN,), dtype='int32')
embedding = Lambda(binarize, output_shape=binarize_outshape)(in_sentence)  # (length * chars, batch)
# embedding = Embedding(num_alphabet, num_alphabet)(in_sentence)
filter_length = [3, 3, 1]
nb_filter = [196, 196, 256]
pool_length = 2
for i in range(len(nb_filter)):
    embedding = Conv1D(filters=nb_filter[i],
                       kernel_size=filter_length[i],
                       padding='valid',
                       activation='relu',
                       kernel_initializer='glorot_normal',
                       strides=1)(embedding)
    embedding = MaxPooling1D(pool_size=pool_length)(embedding)
hidden = Bidirectional(LSTM(
    128, dropout=0.2, recurrent_dropout=0.2))(embedding)
hidden = Dense(128, activation='relu')(hidden)
hidden = Dropout(0.2)(hidden)
output = Dense(num_cat, activation='softmax')(hidden)

model = Model(inputs=in_sentence, outputs=output)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'top_k_categorical_accuracy'])


print('Training...')
BATCH_SIZE = 128
cbTensorBoard = TensorBoard(log_dir='./Graph', histogram_freq=1, 
                            write_graph=True, write_images=False)
model.fit(train_X, train_Y,
          batch_size=BATCH_SIZE,
          epochs=11,
          validation_data=(test_X, test_Y),
          callbacks=[cbTensorBoard])


model.save('p5-40-test.hdf5')

