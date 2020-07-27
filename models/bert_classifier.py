# -*- coding: utf-8 -*-
"""Bert Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fyvk3toLoLfmdUiELHjHWZyMmJ0segCz
"""

!pip install transformers

import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.test.is_gpu_available())

train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
                                  batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)

print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))
train_examples[:2]

train_labels[:2]

from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard


def load_data(examples, targets, tokenizer):

    reviews, labels = [], []
    input_ids, token_type_ids, attention_masks = [], [], []
    for example, label in zip(examples, targets):
      reviews.append(str(example).strip())
      labels.append(label)

    X = tokenizer(reviews, 
                       padding=True, 
                       truncation=True, 
                       return_tensors="tf")

    y = np.array(labels)
    
    print ("begin loding data...")
    data = {}
    data["X"] = X
    data["input_ids"] = X["input_ids"]
    data["token_type_ids"] = X["token_type_ids"]
    data["attention_mask"] = X["attention_mask"]
    data["y"] = y
    data["tokenizer"] = tokenizer
    data["int2label"] =  {0: "negative", 1: "positive"}
    data["label2int"] = {"negative": 0, "positive": 1}

    return data

from transformers import BertTokenizer, TFBertForSequenceClassification

tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
bert = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")

train_data = load_data(train_examples[:5], train_labels, tokenizer)

out = bert(train_data["X"])
out

tf_predictions = tf.nn.softmax(out[0], axis=-1)
tf_predictions

def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

train_data = load_data(train_examples[:100], train_labels, tokenizer)
train_dataset = tf.data.Dataset.from_tensor_slices((train_data["input_ids"], train_data["token_type_ids"], train_data["attention_mask"], train_data["y"])).map(map_example_to_dict).shuffle(32).batch(12)
test_data = load_data(test_examples[:100], test_labels, tokenizer)
test_dataset = tf.data.Dataset.from_tensor_slices((test_data["input_ids"], test_data["token_type_ids"], test_data["attention_mask"], test_data["y"])).map(map_example_to_dict).shuffle(32).batch(12)
train_dataset

learning_rate = 2e-5
number_of_epochs = 5
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
bert.compile(optimizer=optimizer, 
              loss=loss, 
              metrics=[metric])

bert_history = bert.fit(train_dataset, 
                        epochs=number_of_epochs, 
                        validation_data=test_dataset)

!mkdir -p saved_model
bert.save_pretrained('saved_model/my_model')

bert_history

text = "The movie is awesome!"
text = train_data["tokenizer"]([text], padding=True, truncation=True, return_tensors="tf")
prediction = bert(text)

print("Prediction:",  tf.nn.softmax(prediction[0], axis=-1))
print(train_data["int2label"][np.argmax(prediction)])

!ls saved_model/my_model

new_model =  TFBertForSequenceClassification.from_pretrained('saved_model/my_model')
new_model.summary()

text = "The movie is awesome!"
text = train_data["tokenizer"]([text], padding=True, truncation=True, return_tensors="tf")
prediction = new_model(text)

print("Prediction:",  tf.nn.softmax(prediction[0], axis=-1))
print(train_data["int2label"][np.argmax(prediction)])

learning_rate = 2e-5
number_of_epochs = 5
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
new_model.compile(optimizer=optimizer, 
              loss=loss, 
              metrics=[metric])
result = new_model.evaluate(test_dataset)
loss, prediction_scores = result
print('loss is: {}'.format(loss))
print('prediction_scores is: {}'.format(prediction_scores))

