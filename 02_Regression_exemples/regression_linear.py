""" Exemplo de linar regression com pytorch """

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets 
import matplotlib.pyplot as plt

# prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples= 100, n_features=1, noise=20, random_state=1 )

X = torch.from_numpy(X_numpy.astype(np.float32)) # shape: [100,1]
y = torch.from_numpy(y_numpy.astype(np.float32)) # shape: [100] - necessario reshape
y= y.view(y.shape[0],1) # new shape: [100,1]

n_samples, n_features = X.shape

# model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# loss and optimizer
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # model.parameters() = weigths , bias

# training process
n_epochs = 300

for epoch in range(n_epochs):   
    ### forward pass
    # try to predict 
    y_predicted = model(X)
    # compute the loss (error)
    loss = criterion(y, y_predicted)
    
    ### backward pass
    # calculate gradients
    loss.backward() 
    
    ### update weigths
    optimizer.step()
    # empty the gradients for the next epoch
    optimizer.zero_grad()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch+1},  loss {loss.item():.3f} ")
    
# plot regression
y_pred = y_predicted.detach().numpy()
plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, y_pred, "b")
plt.show()
    