# An implementation of linear regression for predicting the price of pizza based on its size
# Implemented by sachin mukherjee
# Training data are - Size - 6,8,10,14,18 and Price - 7,9,13,17.5,18

# Size
X = [[6], [8], [10], [14], [18]]
# Price
Y = [[7], [9], [13], [17.5], [18]]

# Plotting the trainig data using matplot library

# importing the library
import matplotlib.pyplot as plt
# calling method figure
plt.figure()
# setting title for graph
plt.title("Price of Prizza against Size")
# setting x label and y label
plt.xlabel("Price")
plt.ylabel("Size")
# plotting the graph
plt.plot(X, Y, 'k.', color="red")
# setting the axis for graph
plt.axis([0, 25, 0, 25])
# setting grid for graph
plt.grid(True)
# display the graph
plt.show()

# now implementing linear regressiong for predicting price
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, Y)
inp = input("Please Enter Size of Pizza - 6 to 18 ")
y_pred = model.predict(inp)
print "Price of Pizza is" + str(y_pred)