""" Tudo manual, mas descobrindo pesos pelo pytorch"""

import torch

# Exemplo de regressao: f = w * x
# peso correto = 2
# y real = 2 * x

# Dado de treinamento:
X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # peso inicial

# model prediction
def forward(x):
    return w*x

# loss = MSE
def loss( y, y_predicted):
    return (( y_predicted - y)**2).mean()

# gradient
# calculo pelo torch

# Treinamento, descobrir os pesos:
print(f'Prediction before training: f(5) = {forward(5):.3f}')
learning_rate = 0.01
n_iter = 20

for epoch in range(n_iter):
    # predition = foward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients 
    l.backward() #dl/dw
    dw = w.grad
    
    # update weights - essas operacoes nao devem entrar no track do gradiente
    with torch.no_grad():
        w -= learning_rate * dw
        
    # pra nao acumular o gradiente da epoca anterior
    w.grad.zero_()  # _ inplace
    
    if epoch % 2 ==0:
        print(f'epoch {epoch +1}: w = {w:.3f}, loss ={l:.8f}') 

print(f'Prediction after training: f(5) = {forward(5):.3f}')