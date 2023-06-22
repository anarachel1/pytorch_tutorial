""" Exemplo de linar regression com pytorch """

import torch
import numpy as np
import torch.nn as nn
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
bc = datasets.load_breast_cancer()
X , y = bc.data, bc.target

n_samples, n_features = X.shape # 569 , 30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))  # (569,)
y_test = torch.from_numpy(y_test.astype(np.float32)) # (569,)

y_train = y_train.view(y_train.shape[0], 1) # [569,1]
y_test = y_test.view(y_test.shape[0], 1) # [569,1]

# 1) model
# f = wx + b, with sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.lin = nn.Linear(n_input_features, 1)
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.lin(x))
        return y_pred
        
model = LogisticRegression(n_features)

# 2) loss and optimizer
criterion = nn.BCELoss()
learning_rate= 0.1
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

# 3) traing loop
n_epochs = 100
for epoch in range(n_epochs): 
    # try to predict values
    y_pred = model(X_train)
    # look the loss (error)
    loss = criterion(y_pred, y_train)
    # calculate the gradient
    loss.backward()
    # uptade weigths based on grad*lr
    optimizer.step()
    # clean the gradients for next loop
    optimizer.zero_grad()
    
    if epoch % 20 == 0:
        print(f"epoch {epoch+1}, loss {loss.item():.4f}")
        
# 4) evaliation
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = y_pred.round() # classes 0 or 1
    acc = y_pred_class.eq(y_test).sum()/ float(y_test.shape[0])
    print(f"Accuracy on testing: {acc:.4f}")
