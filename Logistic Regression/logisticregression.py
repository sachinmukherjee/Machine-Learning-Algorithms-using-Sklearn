# implementation of logistic regression algorithm using iris data set
# created by sachin mukherjee
# sachinmukherjee29@gmail.com

from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

dataset = datasets.load_iris()

training = dataset.data
label = dataset.target

test = dataset.data[100:150]
expected = dataset.target[100:150]

model = LogisticRegression()
model.fit(training, label)
predicted = model.predict(test)

accuracy = accuracy_score(expected, predicted)
print accuracy * 100

