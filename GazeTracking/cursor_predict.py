from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from tensorflow import keras
from tensorflow.keras import layers

INPUT = pd.read_csv("test_input.csv")
OUTPUT = pd.read_csv("test_output.csv")

OUTPUT.columns = ["outputx", "outputy"]
INPUT.columns = ["lx", "ly", "rx", "ry", "ll1x", "ll1y", "ll2x", "ll2y", "ll3x", "ll3y", "ll4x", "ll4y", "ll5x", "ll5y"
, "ll6x", "ll6y", "rl1x", "rl1y", "rl2x", "rl2y", "rl3x", "rl3y", "rl4x", "rl4y", "rl5x", "rl5y", "rl6x", "rl6y"]

OUTPUT.insert(0, 'id', range(0, len(OUTPUT)))
INPUT.insert(0, 'id', range(0, len(INPUT)))

DATA = pd.merge(INPUT, OUTPUT, on = "id")
DATA = DATA.apply (pd.to_numeric, errors='coerce')
DATA = DATA.dropna()

#DATA = DATA.iloc[:-100].iloc[100:]
del DATA['id']

test_dataset = DATA

# Reset the index
test_dataset = test_dataset.reset_index(drop=True)

test_x_labels = test_dataset.pop("outputx")
test_y_labels = test_dataset.pop("outputy")

test_stats = test_dataset.describe()
test_stats = test_stats.transpose()

def norm(x):
	return (x - test_stats['mean']) / test_stats['std']
normed_test_data = norm(test_dataset)

modelX = keras.models.load_model('predict_xindex.h5')
modelY = keras.models.load_model('predict_yindex.h5')

loss, mae, mse = modelX.evaluate(normed_test_data, test_x_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} x pixels".format(mae))

loss, mae, mse = modelY.evaluate(normed_test_data, test_y_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} y pixels".format(mae))



