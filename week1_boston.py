'''
Created on Jan 29, 2017

@author: derekh1
'''

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.cross_validation  import cross_val_predict

# Load the data from the the boston house-prices dataset 
boston_data = load_boston()
x = boston_data['data']
y = boston_data['target']

#print (type(boston_data))

# Make and fit the linear regression model
# TODO: Fit the model and Assign it to the model variable
model = LinearRegression()
model.fit(x, y)

# Make a prediction using the model
#DGH This is a list of the list of 13 attributes in the Boston dataset
#sample house is a list of lists

sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
print (len(sample_house[0]))
# TODO: Predict housing price for the sample_house
#This is multivariate regression so we are passing each variable into predict function
prediction = model.predict(sample_house)
print (prediction)

#DGH CODE for Practice with matplotlib
lr = LinearRegression()
boston = load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, boston.data, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
if __name__ == '__main__':
    pass