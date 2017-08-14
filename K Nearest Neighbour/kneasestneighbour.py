# Implementation of k nearest neighbour algorithm using iris data set
# created by sachin mukherjee
# sachinmukherjee29@gmail.com

# importing all modules
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# creating training and label from dataset
dataset = datasets.load_iris()
trainging_data = dataset.data
label = dataset.target

# creating our classifier
model = KNeighborsClassifier()
model.fit(trainging_data, label)

test_data = trainging_data[100:150]

predict = model.predict(test_data)
expected = label[100:150]

accuracy = accuracy_score(expected, predict)
print "Accuracy of the classifier is"
print accuracy * 100





