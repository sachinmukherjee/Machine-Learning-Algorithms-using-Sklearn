# implementation of bernoullis naive bayes algorithm for document classification
# created by sachin mukherjee
# sachinmukherjee29@gmail.com

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


training = ['very good movie', 'fucking bad movie', 'imtiaz ali was at its best', 'worst acting by any actor', 'worth buying ticket']
label = ['good', 'bad', 'good', 'bad', 'good']

# conveting training data to vocabulary and counting its frequency measure
count_vect = CountVectorizer()
count_vect.fit(training)
training_count = count_vect.transform(training)
training_count.shape

# now using term frequency times inverse document document frequency

test = ['ranbir kapoor was at in his worst phase of his life watch movie on your own risk']
count_vect.transform(test)


# now running the algorithm

clf = MultinomialNB()
clf.fit(training_count, label)
pred = clf.predict(count_vect.transform(test))
print pred


