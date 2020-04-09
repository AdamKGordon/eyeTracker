from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import functools
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

INPUT = pd.read_csv("input_data.csv")
OUTPUT = pd.read_csv("output_data.csv")

OUTPUT.columns = ["outputx", "outputy"]
INPUT.columns = ["lx", "ly", "ll1x", "ll1y", "ll2x", "ll2y", "ll3x", "ll3y", "ll4x", "ll4y", "ll5x", "ll5y"
, "ll6x", "ll6y", "rx", "ry", "rl1x", "rl1y", "rl2x", "rl2y", "rl3x", "rl3y", "rl4x", "rl4y", "rl5x", "rl5y"
, "rl6x", "rl6y"]

OUTPUT.insert(0, 'id', range(0, len(OUTPUT)))
INPUT.insert(0, 'id', range(0, len(INPUT)))

DATA = pd.merge(INPUT, OUTPUT, on = "id")
DATA = DATA.apply (pd.to_numeric, errors='coerce')
DATA = DATA.dropna()
del DATA['id']

# 60% for training purpose
train_dataset = DATA.sample(frac = 0.6)
# 40% for testing purpose
test_dataset = pd.concat([train_dataset, DATA]).drop_duplicates(keep = False) 

# Reset the index
train_dataset = train_dataset.reset_index(drop=True)
test_dataset = test_dataset.reset_index(drop=True)

train_x_labels = train_dataset.pop("outputx")
train_y_labels = train_dataset.pop("outputy")
test_x_labels = test_dataset.pop("outputx")
test_y_labels = test_dataset.pop("outputy")

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

def norm(x):
	return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# print("train_dataset:")
# print(train_dataset.info)
# print("normed_train_data:")
# print(normed_train_data.info)
# print("train_x_labels:")
# print(test_x_labels)

def build_model():
	model = keras.Sequential([
		layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
		layers.Dense(64, activation='relu'),
		layers.Dense(1)
	])

	optimizer = tf.keras.optimizers.RMSprop(0.001)

	model.compile(loss='mse',
    	optimizer=optimizer,
    	metrics=['mae', 'mse'])
	return model

model = build_model()
# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)
# print(example_result)

class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 10 == 0: print('')
		print('.', end='')

EPOCHS = 100

def plot_Xhistory(history):
	hist = pd.DataFrame(history.history)
	hist['epoch'] = history.epoch

	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [outputx]')
	plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
	plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
	plt.ylim([0,500])
	plt.legend()
	plt.show()

def plot_Yhistory(history):
	hist = pd.DataFrame(history.history)
	hist['epoch'] = history.epoch

	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [outputy]')
	plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
	plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
	plt.ylim([0,500])
	plt.legend()
	plt.show()

model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

historyX = model.fit(normed_train_data, train_x_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

histX = pd.DataFrame(historyX.history)
histX['epoch'] = historyX.epoch
print(histX.tail())

#plot_Xhistory(historyX)

model.save('predict_xindex.h5')

loss, mae, mse = model.evaluate(normed_test_data, test_x_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} x pixels".format(mae))

model = build_model()

historyY = model.fit(normed_train_data, train_y_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

histY = pd.DataFrame(historyY.history)
histY['epoch'] = historyY.epoch
print(histY.tail())

#plot_Yhistory(historyY)

model.save('predict_yindex.h5')

loss, mae, mse = model.evaluate(normed_test_data, test_y_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} y pixels".format(mae))







