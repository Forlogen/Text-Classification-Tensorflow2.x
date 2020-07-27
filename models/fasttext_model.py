# -*- coding: utf-8 -*-
"""FastText Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w6pMODHd4lhjZGgdRx9lasN4eBVT1gY_
"""

import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.test.is_gpu_available())

train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
                                  batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))

train_examples[:2], train_labels[:2]

from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

def load_data(examples, targets, num_words, sequence_length, do_test = False, test_size=0.20, oov_token=None):

    reviews, labels = [], []

    for example, label in zip(examples, targets):
      reviews.append(str(example).strip())
      labels.append(str(label).strip())

    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(reviews)
    X = tokenizer.texts_to_sequences(reviews)
    X, y = np.array(X), np.array(labels)
    X = pad_sequences(X, maxlen=sequence_length)

    # convert labels to one-hot encoded
    y = to_categorical(y)

    
    if do_test == False:
      print ("begin loding training data...")
      X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=1)
      data = {}
      data["X_train"] = X_train
      data["X_val"] = X_val
      data["y_train"] = y_train
      data["y_val"] = y_val
      data["tokenizer"] = tokenizer
      data["int2label"] =  {0: "negative", 1: "positive"}
      data["label2int"] = {"negative": 0, "positive": 1}

      return data
    else:
      print ("begin loding test data...")
      data = {}
      data["X_test"] = X
      data["y_test"] = y
      data["tokenizer"] = tokenizer
      data["int2label"] =  {0: "negative", 1: "positive"}
      data["label2int"] = {"negative": 0, "positive": 1}

    return data

class Classifier:

  def __init__(self, word_index, embedding_dims, max_len, num_class = 2):

    self.vocab_size = len(word_index) + 1
    self.embedding_dims = embedding_dims
    self.max_len = max_len
    self.num_class = num_class

  def create_model(self):
    model = Sequential([
        tf.keras.layers.Embedding(self.vocab_size, self.embedding_dims, input_length = self.max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(self.num_class),
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
    
    return model

embedding_dims = 300
max_len= 100

data = load_data(train_examples, train_labels, 10000, 100, do_test = False)

classifier = Classifier(data["tokenizer"].word_index, embedding_dims, max_len)

model = classifier.create_model()
model.summary()

if not os.path.isdir("logs"):
    os.mkdir("logs")

tensorboard = TensorBoard(log_dir=os.path.join("logs", "IMDB"))

history = model.fit(data["X_train"], data["y_train"],
                    batch_size=256,
                    epochs=10,
                    validation_data=(data["X_val"], data["y_val"]),
                    callbacks=[tensorboard],
                    verbose=1)

model.save("IMDB.h5", overwrite=True, include_optimizer = True, save_format="tf")

new_model = tf.keras.models.load_model("IMDB.h5")
new_model.summary()

text = "The movie is awesome!"
sequence = data["tokenizer"].texts_to_sequences([text])
# pad the sequences
sequence = pad_sequences(sequence, maxlen=max_len)
prediction = new_model.predict(sequence)[0]
print(prediction)
print(data["int2label"][np.argmax(prediction)])

