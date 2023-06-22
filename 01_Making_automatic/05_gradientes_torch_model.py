""" Automatizando TUDO - agora a function"""

import torch
import torch.nn as nn

# Exemplo de regressao: f = w * x
# peso correto = 2
# y real = 2 * x

# Dado de treinamento:
# novo shape : modelo aceita 2D onde nRows = nSamples e nColumns= nFeatures
X = torch.tensor([[1], [2], [3], [4]], dtype = torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype = torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init___()
        # define layers
        self.lin = nn.Lienar(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

X_test = torch.tensor([5], dtype=torch.float32)

# Treinamento, descobrir os pesos:
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')
learning_rate = 0.01
n_iter = 20

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iter):
    # predition = foward pass
    y_pred = model(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients 
    l.backward() #dl/dw
    
    # update weights 
    optimizer.step()
        
    # pra nao acumular o gradiente da epoca anterior
    optimizer.zero_grad()  
    
    if epoch % 2 ==0:
        [w, b] = model.parameters()
        print(f'epoch {epoch +1}: w = {w[0][0].item():.3f}, loss ={l:.8f}') 

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')