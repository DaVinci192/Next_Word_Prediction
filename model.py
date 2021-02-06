# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:18:21 2021

@author: Randhir
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np 
import os
import random
import math
import tarfile

# Tokenization and savingof tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle   

# transform dependent data to categorical data
from tensorflow.keras.utils import to_categorical

# building and testing the model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

# Getting the Universal Sentence Encoder
directory = #r'Path to model folder' or from url:  https://tfhub.dev/google/universal-sentence-encoder/4

for fname in os.listdir(directory):
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()

loaded_model = tf.keras.models.load_model(directory)


# dataset: https://www.kaggle.com/jannesklaas/scifi-stories-text-corpus
file = open("data/internet_archive_scifi_v3.txt")

sentences = []
current_sentence = []

def convert_to_string(s): 
    string = "" 
    return string.join(s)

# reading the data and cleaning of all special characters:
while 1:
    tmp_char = file.read(1)
    tmp_ascii = ord(tmp_char)
    
    if (((tmp_ascii >= ord('A') and tmp_ascii <= ord('Z')) or (tmp_ascii >= ord('a') and tmp_ascii <= ord('z'))) or (tmp_ascii == 39)):
        word_c = []
        while (((tmp_ascii >= ord('A') and tmp_ascii <= ord('Z')) or (tmp_ascii >= ord('a') and tmp_ascii <= ord('z'))) or (tmp_ascii == 39)):
            word_c.append(chr(tmp_ascii))
            tmp_ascii = ord(file.read(1))
        word = convert_to_string(word_c)
        current_sentence.append(word)
    
    if (tmp_ascii == ord('.')):
        sentences.append(current_sentence)
        current_sentence = []
    
file.close()


# generating indepent and dependent data 
X = []
y = []

def convert_to_sentence(w):
    space = " "
    return space.join(w)

MIN_INPUT_SIZE = 3 # minimum sentence. may be increased or decreased to change model performance
test_size = math.floor(len(sentences)*0.01/100) # .01% of the data, small sample used for testing (may be increased for better accuracy)

test_sentence = []

for sentence in range(0, test_size):
    tmp_sentence = sentences[sentence]
    
    if len(tmp_sentence) > MIN_INPUT_SIZE:
        test_sentence.append(tmp_sentence)
        INPUT_SIZE = random.randint(MIN_INPUT_SIZE, len(tmp_sentence) - 1)
            
        
        for shift in range(0, len(tmp_sentence) - INPUT_SIZE):
            x_arr = []
            x_str = " "
            y_str = tmp_sentence[INPUT_SIZE + shift]
            for word in range(0, INPUT_SIZE):
                x_arr.append(tmp_sentence[word + shift])
            x_str = convert_to_sentence(x_arr)
            X.append(loaded_model([x_str]))
            y.append(y_str)
        
        progress = sentence * 100 / test_size 
        print("Generating Dataset... {}%".format(progress)) # printing progress to get status of dataset downloaad

sentences = test_sentence

 
# tokenizing the dependent variable
tokenizer = Tokenizer()
pickle.dump(tokenizer, open('tokenizer1.pkl', 'wb'))

text = " "

def convert_to_full_text(s):
    text_sentences = []
    for sentence in range(0, len(s)):
        text_sentences.append(convert_to_sentence(s[sentence]))
        
    space = " "
    return space.join(text_sentences)

text = convert_to_full_text(sentences) 
tokenizer.fit_on_texts([text])
y_text = convert_to_sentence(y)
y = tokenizer.texts_to_sequences([y_text])

# formatting data as numpy array
X = np.array(X)
y = np.array(y)

vocab_size = 0

for sentence in range(0, len(sentences)):
    for word in range(0, len(sentences[sentence])):
        vocab_size += 1
    
# reformatting data - ISSUE: unsure how to make y and vocab the same size while preserving data
y = to_categorical(y, num_classes=vocab_size, dtype='float32')
X = np.swapaxes(X, 0, 1)

y = np.swapaxes(y, 1, 2)
y = y[:,:,0]

# building and testing the model - structure taken from https://towardsdatascience.com/next-word-prediction-with-nlp-and-deep-learning-48b9fe0a17bf
# CODE NOT ABLE TO BE TESTED
model = Sequential()
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(512))
model.add(Dense(512, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))


checkpoint = ModelCheckpoint("nextword1.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')

reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

logdir='logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))
model.fit(X, y, epochs=150, batch_size=64, callbacks=[checkpoint, reduce, tensorboard_Visualization])



