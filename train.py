import re
import os
import sys
import json
import time
import string
import pickle
import models
import statistics
from keras.models import load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.losses import categorical_crossentropy
import numpy as np
np.random.seed(1234)
import warnings
warnings.filterwarnings("ignore") #ignore warnings

import settings

padding_type = settings.padding_type
truncating_type = settings.truncating_type
max_length = settings.max_length
dictionary_length = settings.dictionary_length

with open("sentences.pkl", 'rb') as f:
	train_sentences = pickle.load(f, encoding='latin1')

with open("indexed_sentences.pkl",'rb') as f:
	corpus = pickle.load(f, encoding = 'latin1')

with open("indexed_labels.pkl", 'rb') as f:
	labels = pickle.load(f, encoding = 'latin1')

x_train = corpus
y_train = labels

#padding the input sequences
padded_seq = pad_sequences(x_train, padding = padding_type, truncating= truncating_type, maxlen=max_length)

batch_time = 0
epoch = settings.epoch
step = settings.step
batch_size = settings.batch_size
loss_array = []
check_point_loss = 9999

model = models.LSTM_model()

if os.path.isfile("model_weights.h5"):
    try:
        model.load_weights("model_weights.h5")
        print("Loaded previously trained weights!")
    except:
        pass

try:
    epoch = int(sys.argv[1])
except:
    pass

loss = []
print("Number of Batches for each epoch: {}".format(int(len(x_train)/step)))
for k in range(epoch):
    print('-'*80)
    print('Epoch {}/{}'.format(k,epoch))
    print('-'*80)
    batch = 0
    tim = time.time()
    temp_loss = []

    for i in range(0, len(x_train),step):
        batch += 1
        tic = time.time()

        x_seq = padded_seq[i:i+step]
        y_encoded = np.eye(dictionary_length)[np.array(y_train[i:i+step]).astype('int')]

        history = model.fit(x_seq, y_encoded, batch_size=batch_size, epochs=1, verbose = 0)
        batch_time += time.time() - tic
        
        temp_loss.append(history.history['loss'][-1])
        if batch%10 == 0:
            print((" "*5+"Batch {}/{} ----- Loss: {:.5f} ------ Time: {:.3f}s").format(batch,int(len(x_train)/step),history.history['loss'][-1], batch_time ))
            batch_time = 0
            
    epoch_loss = statistics.mean(temp_loss)
    if check_point_loss > epoch_loss:
    	check_point_loss = epoch_loss
    	print("Loss for epoch: ", epoch_loss)
    	print("Model weights is saving to model_weights.h5")
    	model.save_weights('model_weights.h5')

    loss.append(epoch_loss)
    loss_array.append(epoch_loss)
    print((" "*38+"Time for Epoch: {:.3f}s").format(time.time()-tim))

plt.plot(loss)

