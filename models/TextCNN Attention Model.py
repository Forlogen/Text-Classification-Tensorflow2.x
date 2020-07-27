#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())


# In[2]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.test.is_gpu_available())


# In[3]:


train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
                                  batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)


# In[4]:


print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))
train_examples[:2]


# In[5]:


train_labels[:2]


# In[6]:


from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard


def load_data(examples, targets, num_words, sequence_length, test_size=0.20, oov_token=None):

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
    
    print ("begin loding data...")
    data = {}
    data["X_train"] = X
    data["y_train"] = y
    data["tokenizer"] = tokenizer
    data["int2label"] =  {0: "negative", 1: "positive"}
    data["label2int"] = {"negative": 0, "positive": 1}

    return data


# In[7]:


class TextCNNAttention(tf.keras.Model):
    def __init__(self,
                 word_index,
                 embedding_dims,
                 maxlen,
                 class_num=2,
                 weights=None,
                 weights_trainable=False,
                 kernel_sizes=[3, 4, 5],
                 filter_size=128,
                 name=None,
                 **kwargs):
      
        super(TextCNNAttention, self).__init__(name=name, **kwargs)

        self.vocab_size = len(word_index) + 1
        self.max_len = max_len
        self.kernel_sizes = kernel_sizes

        if weights != None:
            weights = np.array(weights)
            self.embedding = tf.keras.layers.Embedding(self.vocab_size, 
                                  embedding_dims,input_length=self.max_len, 
                                  weights=[weights],
                                  trainable=weights_trainable)
        else:
            self.embedding = tf.keras.layers.Embedding(self.vocab_size, 
                                  embedding_dims,
                                  input_length=self.max_len)

        self.convs = []
        self.max_poolings = []
        for i, k in enumerate(kernel_sizes):
            self.convs.append(tf.keras.layers.Conv1D(filter_size, k, activation="relu"))
            self.max_poolings.append(tf.keras.layers.GlobalAvgPool1D())
        self.dense = tf.keras.layers.Dense(class_num, activation='softmax')
        self.bn = tf.keras.layers.BatchNormalization()
        self.attention = tf.keras.layers.Attention()

    def call(self, inputs, training=True):
        q_embed = self.embedding(inputs)
        v_embed = self.embedding(inputs)
        convs = []

        for i, k in enumerate(self.kernel_sizes):
            q = self.convs[i](q_embed)
            v = self.convs[i](v_embed)

            q = self.max_poolings[i](q)
            v = self.max_poolings[i](v)
            q_v = self.attention([q, v])

            convs.append(q_v)

        out = tf.keras.layers.concatenate(convs)

        out = self.bn(out, training=training)

        out = self.dense(out)

        return out


# In[8]:


embedding_dims = 300
max_len= 100
filter_size = 2


data = load_data(train_examples[:], train_labels, 10000, 100)

model = TextCNNAttention(data["tokenizer"].word_index, embedding_dims, max_len, filter_size)



if not os.path.isdir("logs"):
    os.mkdir("logs")

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[9]:


if not os.path.isdir("logs"):
    os.mkdir("logs")
tensorboard = TensorBoard(log_dir=os.path.join("logs", "IMDB"))

history = model.fit(data["X_train"], data["y_train"],
                    batch_size=256,
                    epochs=10,
                    validation_split = 0.1,
                    callbacks=[tensorboard])

model.save_weights("IMDB.h5", overwrite=True)
model.summary()


# In[11]:


def get_predictions(text):
    sequence = data["tokenizer"].texts_to_sequences([text])
    # pad the sequences
    sequence = pad_sequences(sequence, maxlen=100)
    # get the prediction
    prediction = model.predict(sequence)[0]
    return prediction, data["int2label"][np.argmax(prediction)]

text = "The movie is awesome!"
output_vector, prediction = get_predictions(text)
print("Output vector:", output_vector)
print("Prediction:", prediction)


# In[14]:


test_data = load_data(train_examples, train_labels, 10000, 100)

new_model = TextCNNAttention(test_data["tokenizer"].word_index, embedding_dims, max_len, filter_size)
new_model.load_weights("IMDB.h5", by_name=True)


# In[15]:


text = "The movie is awesome!"
sequence = test_data["tokenizer"].texts_to_sequences([text])
sequence = pad_sequences(sequence, maxlen=100)
prediction = new_model.predict(sequence)[0]

print(prediction)
print(test_data["int2label"][np.argmax(prediction)])


# In[ ]:




