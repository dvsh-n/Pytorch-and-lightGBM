# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 23:14:46 2020

@author: olhartin@asu.edu
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
4)
"""
import torch
import numpy as np

## sequential neural net
## torch loss function
## simple example random inputs, and initial weights
## gradients and do gradient descents

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs +++and outputs
x = torch.randn(N, D_in)
##  normalize/standardization
x -= x.mean(1,keepdim=True)
x /= x.std(1,keepdim=True)

y = torch.randn(N, D_out)

def onehtar(y):
    for i in range(len(y[:,0])):
        maxval = torch.max(y[i,:])  ## if you use maxval = np.max(y[i,:]) this will work for numpy array
        for j in range(len(y[0,:])):
            if (y[i,j] == maxval):
                y[i,j] = 1.0
            else:
                y[i,j] = 0.0
#        print('cat',y[i,:],'maxval',maxval)
    return(y)
y = onehtar(y)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-2
last = 1.0
error = 1.0
t = 0
while (t<5000 and error>0.000001):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()
    error = abs(last-loss.item())
    last = loss.item()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
        #
    t += 1
##
y_pred = onehtar(y)            
print(y_pred)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y, y_pred)
print('Dataset Accuracy: %.2f' % acc)
print('Misclassified samples: %d' % (y != y_pred).sum())
