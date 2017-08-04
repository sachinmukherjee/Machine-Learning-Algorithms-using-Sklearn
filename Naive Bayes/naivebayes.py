# Implementation of naive bayes classifier on iris data set
# created by sachin mukherjee
# sachinmukherjee29@gmail.com

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

dataset = datasets.load_iris()
model = GaussianNB()

# Training model
training_example = dataset.data
label = dataset.target

# Testing
test_set = dataset.data[100:150]
expected = dataset.target[100:150]

model = GaussianNB()
model.fit(training_example, label)
print "Excepted Result"
print expected
print "\n"
print "Predicted Result"
predicted = model.predict(test_set)
print predicted

accuracy = accuracy_score(expected, predicted)
print accuracy * 100