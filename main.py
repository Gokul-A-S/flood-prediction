

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers

raw_dataset=pd.read_csv('test.csv')
print(tf.__version__)
dataset=raw_dataset.copy()
dataset.head()

dataset.drop('YEAR',inplace=True,axis=1)
dataset.drop('MONTHLY RAINFALL',inplace=True,axis=1)

print(dataset)

train_dataset=dataset.sample(frac=0.8,random_state=0)
test_dataset=dataset.drop(train_dataset.index)
print(test_dataset)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Last 10 Year Average (Flow in cumecs)')
test_labels = test_features.pop('Last 10 Year Average (Flow in cumecs)')

train_dataset.describe().transpose()[['mean', 'std']]

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu',input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


history = dnn_model.fit(
     train_features,
     train_labels,
     validation_split=0.2,
     verbose=0, epochs=500)

import matplotlib.pyplot as plt
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [R]')
  plt.legend()
  plt.grid(True)
plot_loss(history)

print(dnn_model.evaluate(test_features, test_labels, verbose=0))

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [R]')
plt.ylabel('Predictions [R]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
#print(test_predictions)

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [R]')
_ = plt.ylabel('Count')

dnn_model.predict([142.53,6.10,41.01,1.17,31.31,19.47])












