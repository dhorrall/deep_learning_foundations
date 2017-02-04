'''
Created on Feb 2, 2017

@author: derekh1
'''
import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Make some fake data
X = np.random.randn(4)

#This is creating the 4 x 3 matrix holding weights that connect the 4 input nodes to the 3 hidden nodes
weights_in_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))

#This is creating a 3 x 2 matrix holding weights that connect the 3 hidden nodes to the 2 output nodes
weights_hidden_out = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))


# TODO: Make a forward pass through the network

##Take the dot product of the 4 random inputs x the 4 X 3 Weight matrix resulting in a size 3 array representing the 3 hidden nodes
hidden_layer_in = np.dot(X, weights_in_hidden)
print('Hidden-layer Input:')
print(hidden_layer_in)

#Run activation function on the result of the matrix multiplication
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

##Take the dot product of the 3 hidden layer outputs x the 3 X 2 Weight matrix resulting in a size 2 array representing the 2 output nodes
output_layer_in = np.dot(hidden_layer_out, weights_hidden_out)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)