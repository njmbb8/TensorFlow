from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def input_fn(features, labels, training=True, batch_size=256):
    #convert inputs to a dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    #shuffle and repeat if in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names = CSV_COLUMN_NAMES, header = 0)
test = pd.read_csv(test_path, names = CSV_COLUMN_NAMES, header = 0)

#print(test.head()) at this point will show a table of the first five rows
#with the species displaying as numbers

train_y = train.pop("Species")
test_y = test.pop("Species")

#feature columns describe how to use the input
#essentially like defining variables with names and dtatypes, etc for each feature

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#building a Deep Neural Network(DNN) with 2 hidden layers with 30 and 10 nodes each
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30,10], #two hiddwn layers of 30 and 10 nodes, respectively
    n_classes=3) #3 because there are three classes(in this case, species)

classifier.train(
    input_fn = lambda: input_fn(train, train_y, training = True),
    steps = 5000) #train until 5000 items have been looked at

eval_result = classifier.evaluate(
    input_fn = lambda: input_fn(test, test_y, training = False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

def input_fn(features, batch_size=256):
    #convert inputs to a data set without labels because the label is being predicted
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted")
for feature in features:
    valid = True
    val = ""
    while valid:
        val = input(feature +": ")
        if not val.isdigit(): valid = False

    predict[feature] = [float(val)]

predictions = classifier.predict(Input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100*probability))
