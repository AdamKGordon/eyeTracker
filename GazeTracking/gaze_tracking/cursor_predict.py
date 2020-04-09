from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
import random

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

modelX = keras.models.load_model('predict_xindex.h5')
modelY = keras.models.load_model('predict_yindex.h5')

random_index = random.randint(0, normed_test_data.shape[0])

predictX = modelX.predict(normed_test_data.loc[[random_index]]).flatten().tolist()[0]
predictY = modelY.predict(normed_test_data.loc[[random_index]]).flatten().tolist()[0]

actualX = test_x_labels.loc[[random_index]].tolist()[0]
actualY = test_y_labels.loc[[random_index]].tolist()[0] 

print("predictX: " + str(predictX))
print("actualX: " + str(actualX))
print("predictY: " + str(predictY))
print("actualY: " + str(actualY))

# loss, mae, mse = modelX.evaluate(normed_test_data, test_x_labels, verbose=2)
# print("Testing set Mean Abs Error: {:5.2f} x pixels".format(mae))
# loss, mae, mse = modelY.evaluate(normed_test_data, test_y_labels, verbose=2)
# print("Testing set Mean Abs Error: {:5.2f} y pixels".format(mae))

#


