# An implementation of multiple linear regression for predicting price of pizza based on its size and number of toplings
# created by sachin mukherjee
# Size and Toplings X= [[6, 2], [8, 1], [10, 0], [14, 2], [1, 18, 0]]
# Price Y = [[7], [9], [13], [17.5], [18]]

# Now implementing Multiple Linear Regression

from sklearn.linear_model import LinearRegression

X= [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
Y = [[7], [9], [13], [17.5], [18]]

model = LinearRegression()
model.fit(X, Y)
prediction = model.predict([8, 2])
print "Price of Pizza is" + str(prediction)