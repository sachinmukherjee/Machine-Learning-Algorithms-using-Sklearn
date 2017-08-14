# Implementation of decision tree algorithm using iris data set
# created by sachin mukherjee
# sachinmukherjee29@gmail.com

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

dataset = datasets.load_iris()
training = dataset.data
label = dataset.target

test = dataset.data[100:120]
expected = dataset.target[100:120]

model = DecisionTreeClassifier()
model.fit(training, label)
predict = model.predict(test)

accuracy = accuracy_score(expected, predict)
print accuracy * 100

