# !pip install music21 
# !apt-get install -y lilypond

# Commented out IPython magic to ensure Python compatibility.
# load libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# %matplotlib inline
from sklearn.model_selection import train_test_split

import music21
from music21 import *

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adamax

import json
import random
import pickle
import IPython
from IPython.display import Image, Audio

# load corpus
with open("/content/drive/MyDrive/ML PROJECT/mozart-corpus.txt", 'r') as f:
  corpus = json.load(f)

"""### Create a list of sorted unique characters"""

# store all the unique characters in the corpus to a mapping dictionary
symbol = sorted(list(set(corpus)))

corpus_len = len(corpus)
symbol_len = len(symbol)

# build a dictionary to access the vocabulary from indices & vice versa
map = dict((c, i) for i, c in enumerate(symbol))
reverse_map = dict((i, c) for i, c in enumerate(symbol))

print("Total number of characters:", corpus_len)
print("Number of unique characters:", symbol_len)

"""### Encode & split the corpus as labels & targets"""

length = 40
features = []
targets = []

for i in range(0, corpus_len - length, 1):
  feature = corpus[i:i+length]
  target = corpus[i+length]
  features.append([map[j] for j in feature])
  targets.append(map[target])

datapoints_len = len(targets)
print("Total number of sequences in the corpus:", datapoints_len)

# reshape X & normalize
X = (np.reshape(features, (datapoints_len, length, 1))) / float(symbol_len)

# one hot encode the output variable
y = tf.keras.utils.to_categorical(targets)

"""Splitting train & seed datasets"""

X_train, X_seed, y_train, y_seed = train_test_split(X, y, test_size = 0.2, random_state = 42)

# hyperparameters
rate = 0.1
learning_rate = 0.01
batch_size = 256
epochs = 200
optimizer = Adamax(learning_rate = learning_rate)
loss = 'categorical_crossentropy'
metrics = ['accuracy']

"""### MODEL"""

model = Sequential()
model.add(LSTM(512, input_shape = (X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(rate = rate))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(rate = rate))
model.add(Dense(y.shape[1], activation = 'softmax'))

model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
model.summary()

# train the model
history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs) # validation_data = X_seed

# save the model
model.save("/content/drive/MyDrive/ML PROJECT/mozart-model.h5")

# save training history
with open('/content/drive/MyDrive/ML PROJECT/history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# load the model
from tensorflow import keras
model = keras.models.load_model('/content/drive/MyDrive/ML PROJECT/mozart-model.h5')

# # load training history
with open('/content/drive/MyDrive/ML PROJECT/history', "rb") as file_pi:
    history = pickle.load(file_pi)

# # Note: to reload, just use history instead of history.history

"""### Evaluate Model"""

# plot the learning curve for accuracy
history_df = pd.DataFrame(history.history)
fig = plt.figure(figsize = (10, 5))
fig.suptitle("Learning Plot of Model for Accuracy")
pa = sns.lineplot(data = history_df["accuracy"], color="orange")
pa.set(xlabel = "Epochs")
pa.set(ylabel = "Training Accuracy")

# plot the learning curve for loss
history_df = pd.DataFrame(history.history)
fig = plt.figure(figsize = (10, 5))
fig.suptitle("Learning Plot of Model for Loss")
pl = sns.lineplot(data = history_df["loss"], color="orange")
pl.set(xlabel = "Epochs")
pl.set(ylabel = "Training Loss")

# function to display music sheet
def show(music):
    display(Image(str(music.write("lily.png"))))
    
def chords_n_notes(snippet):
    melody = []
    offset = 0
    for i in snippet:
        # if it is chord
        if "." in i or i.isdigit():
            # seperate the notes in chords
            chord_notes = i.split(".")
            notes = []
            for j in chord_notes:
                inst_note = int(j)
                note_snip = note.Note(inst_note)
                notes.append(note_snip)
                chord_snip = chord.Chord(notes)
                chord_snip.offset = offset
                melody.append(chord_snip)
        # if it is a note
        else:
            note_snip = note.Note(i)
            note_snip.offset = offset
            melody.append(note_snip)
        # increase offset each iteration so that notes do not stack
        offset += 1
    
    melody_midi = stream.Stream(melody)
    return melody_midi

# function to generate music
def melody_generator(note_count):
  seed = X_seed[np.random.randint(0, len(X_seed) - 1)]
  music = ""
  notes = []
  for i in range(note_count):
    seed = seed.reshape(1, length, 1)
    prediction = model.predict(seed, verbose = 0)[0]
    prediction = np.log(prediction) / 1.0
    exp_preds = np.exp(prediction)
    prediction = exp_preds/np.sum(exp_preds)

    index = np.argmax(prediction)
    index_N = index/float(symbol_len)
    notes.append(index)
    music = [reverse_map[char] for char in notes]

    seed = np.insert(seed[0], len(seed[0]), index_N)
    seed = seed[1:]

  melody = chords_n_notes(music)
  midi = stream.Stream(melody)
  return music, midi

music_notes, melody = melody_generator(100)
show(melody)

# save the generated music
melody.write("midi", "deep-mozart-3.mid")

"""Convert midi to wav outside of this notebook.  <br>
https://audio.online-convert.com/convert/midi-to-wav
"""

# play generated music
IPython.display.Audio("/content/drive/MyDrive/ML PROJECT/deep-mozart-2.wav")