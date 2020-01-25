
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import settings
import pickle
import text_cleaning
import models

def get_key(val): 
    for key, value in word_index.items():
        if val == value:
            return key

def get_answer(text:str, word_index, padding_type:str, truncating_type:str, max_length:int):
    index_array = []
    text = text_cleaning.clean_text(text)

    for i in text.split():
        index = word_index.get(i)
        if index == None:
            index_array.append(word_index.get('<OOV>'))
        else:
            index_array.append(index)

    index_array = pad_sequences(np.array([index_array]), padding=padding_type, truncating=truncating_type, maxlen=max_length)[0]
    val_pred = model.predict(np.array([index_array]))[0]
    index_values = val_pred.argsort()[-10:][::-1]

    t = 0
    words = []
    while len(words)!=3:
        word = get_key(index_values[t])
        if word != "<OOV>":
            words.append(word)
        t+=1
    return ", ".join(words)

with open("word_index_dic.pkl", 'rb') as f:
    word_index = pickle.load(f, encoding='latin1')

model = models.LSTM_model()
model.load_weights("model_weights.h5")

for text in settings.test_samples:
    print()
    preds = get_answer(text, word_index, padding_type = settings.padding_type, truncating_type = settings.truncating_type, max_length = settings.max_length)
    print("Sentence : {} ____".format(text))
    print("Top words: "+preds)
    print()
