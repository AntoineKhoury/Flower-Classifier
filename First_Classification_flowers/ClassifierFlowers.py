import numpy as np 
import pandas as pd 
import tensorflow as tf 


####### Data processing

## Import Data from file using pandas
data_pd = pd.read_csv("plantdata.txt", sep = ',', names = ['Feature_1','Feature_2', 'Feature_3','Feature_4', 'Type'])

## Normalize data
cols_to_norm = ['Feature_1','Feature_2', 'Feature_3','Feature_4']
data_pd[cols_to_norm] = data_pd[cols_to_norm].apply(lambda x: (x - x.min())/ (x.max() - x.min()) )

## Map labels with integers
data_pd['Type'] = data_pd['Type'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2})

## Assign arrays to input and output of data
x_data = data_pd.drop('Type', axis = 1)
y_data = data_pd['Type']

## Split between train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 100)




############# Models


## Create features of the model
feature_1 = tf.feature_column.numeric_column('Feature_1')
feature_2 = tf.feature_column.numeric_column('Feature_2')
feature_3 = tf.feature_column.numeric_column('Feature_3')
feature_4 = tf.feature_column.numeric_column('Feature_4')


### Create feature list
features = [feature_1,feature_2,feature_3,feature_4]


## Define input function

input_func = tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=1,num_epochs=1000, shuffle= True)



##### Linear Classifier

# n_classes is number of output to classify between
linear_model = tf.estimator.LinearClassifier(feature_columns = features, n_classes = 3)
linear_model.train(input_fn=input_func, steps = 500)


# Evaluate the performance of our model
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size = 10, num_epochs=1, shuffle = False)
print("Results for Linear Classifier")
result = linear_model.evaluate(eval_input_func)
print(result)



## See what is predicted for our test values
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test, batch_size = 10, num_epochs=1, shuffle=False)
predictions = linear_model.predict(pred_input_func)

my_pred= list(predictions)
label_predicted = [p['class_ids'][0] for p in my_pred]
y_test_array = y_test.values
confusion_matrix = tf.confusion_matrix(y_test_array, label_predicted)

with tf.Session():
    print('\nConfusion Matrix:\n', tf.Tensor.eval(confusion_matrix,feed_dict=None, session=None))




###### Deep Neural Network model

## Train model
dnn_model = tf.estimator.DNNClassifier(hidden_units = [10,10,10], feature_columns = features, n_classes = 3)
dnn_model.train(input_fn= input_func, steps = 500)

# Evaluate the performance of our model
eval_input_function = tf.estimator.inputs.pandas_input_fn(x=x_test, y = y_test, batch_size = 10, num_epochs= 1, shuffle = False)
print("Results for DNN Classifier")
result = dnn_model.evaluate(eval_input_function)
print(result)


## See what is predicted for our test values
predictions = dnn_model.predict(pred_input_func)
my_pred= list(predictions)
label_predicted = [p['class_ids'][0] for p in my_pred]
y_test_array = y_test.values
confusion_matrix = tf.confusion_matrix(y_test_array, label_predicted)

with tf.Session():
    print('\nConfusion Matrix:\n', tf.Tensor.eval(confusion_matrix,feed_dict=None, session=None))

