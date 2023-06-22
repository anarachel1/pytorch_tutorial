""" Como descobrir pesos manualmente """

import numpy as np

# Exemplo de regressao: f = w * x
# peso correto = 2
# y real = 2 * x

# Dado de treinamento:
X = np.array([1, 2, 3, 4], dtype = np.float32)
Y = np.array([2, 4, 6, 8], dtype = np.float32)

w = 0.0  # peso inicial

# model prediction
def forward(x):
    return w*x

# loss = MSE
def loss( y, y_predicted):
    return (( y_predicted - y)**2).mean()

# gradient
# MSE = 1/n * (w*x - y) **2
# dJ/dw = 1/n 2x (w*x - y) 
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

# Treinamento, descobrir os pesos
print(f'Prediction before training: f(5) = {forward(5):.3f}')
learning_rate = 0.01
n_iter = 20

for epoch in range(n_iter):
    # predition = foward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients 
    dw = gradient(X, Y, y_pred)
    
    # update weights
    w -= learning_rate * dw
    
    if epoch % 2 ==0:
        print(f'epoch {epoch +1}: w = {w:.3f}, loss ={l:.8f}') 

print(f'Prediction after training: f(5) = {forward(5):.3f}')