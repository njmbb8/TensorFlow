from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves  import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

def make_input_fn(data_df, label_df, num_epochs = 10, shuffle = True, batch_size=32):
    def input_function(): #inner function, gets returned
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) #create a tf.data.Dataset object and its label
        if shuffle:
            ds = ds.shuffle(1000) #randomize order of data
        ds = ds.batch(batch_size).repeat(num_epochs) #split data into batches and repeat for epoch times
        return ds #returns a single batch
    return input_function #returns a function object


dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #training dataset
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #testing data
y_train = dftrain.pop('survived') #pop removes a specific column and stores it in the variable
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch',
                       'class', 'deck', 'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() #gets all unique entries in a feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns) #linear estimator uses feature column to create model

linear_est.train(train_input_fn) #train
result = linear_est.evaluate(eval_input_fn) #get model metrics/stats by testing on testing data

clear_output() #clears console output
print(result['accuracy']) #the result variable is simply a dict of stats about our model

result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[1])
print(result[1]['probabilities'][0])
