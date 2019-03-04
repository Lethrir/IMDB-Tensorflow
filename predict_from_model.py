from __future__ import absolute_import, division, print_function

import os
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

import numpy as np

# review_text = "dreadfull film"
# review_text = "this movie is not awful it could just do with being better"
# review_text = "terrible terrible film dreadfull acting and dodgy script"
review_text = "loved it"
# get dataset
imdb = keras.datasets.imdb

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index(path="imdb_word_index.json")

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

def encode_review(text):
    first_pass = [word_index[x] for x in text.split(' ')]
    return np.pad(first_pass, (0,256-len(first_pass)), 'constant', constant_values=(word_index["<PAD>"]))

encoded = encode_review(review_text)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

prediction = model.predict(np.array([encoded]))[0][0]
prediction_class = model.predict_classes(np.array([encoded]))[0][0]
print("Prediction:")
print(prediction)
print("Prediction class:")
print(prediction_class)

if prediction_class == 1:
    print('Good')
else:
    print("Bad")
