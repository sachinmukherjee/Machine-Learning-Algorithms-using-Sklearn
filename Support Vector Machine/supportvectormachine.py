# Implementation of support vector machine algorithm using iris dataset
# created by sachin mukherjee
# sachinmukherjee29@gmail.com

from sklearn import svm
from sklearn import datasets
from sklearn.metrics import accuracy_score
dataset = datasets.load_iris()

training_example = dataset.data
label = dataset.target

test = dataset.data[100:150]
expected = dataset.target[100:150]

clf = svm.SVC(kernel="linear")
clf.fit(training_example, label)
predicted = clf.predict(test)

accuracy  = accuracy_score(expected, predicted)
print accuracy * 100