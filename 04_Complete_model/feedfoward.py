import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
input_size = 784 #28x28 pixels
hidden_size = 100
num_classes = 10 # 10 differents digits to predict
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST data
train = torchvision.datasets.MNIST(root='./data', train=True, 
                                    transform=transforms.ToTensor(), download=True)

test = torchvision.datasets.MNIST(root='./data', train=False, 
                                    transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader( dataset=train, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader( dataset=test, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)  # for older pytorch versions use: examples.next()
print(samples.shape, labels.shape)  # samples: [batch, channel =1, pixels, pixels ]

'''
# visu some digits in the first batch
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap = 'gray')
plt.savefig('fig.jpg')
'''
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no softmax because of crossEntropyLoss
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate )

# training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # sample: 100,1,28,28 needs to flatten it, so model input = 100, 28*28 
        images = images.reshape(-1, 28*28).to(device)
        labels.to(device)
        
        #foward
        outputs = model(images)  #prediction
        loss = criterion(outputs, labels) # calculate error (loss)
        
        #backward
        optimizer.zero_grad() # clean
        loss.backward() # calculate gradients
        optimizer.step() # adjust the weights
        
        if (i+1)% 100 ==0:
            print(f'epoch {epoch+1}/{num_epochs} , step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels.to(device)
        outputs = model(images)
        
        _, predictions = torch.max(outputs, 1) # value, index
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    
    acc = 100* n_correct / n_samples
    print(f'Accuracy = {acc}%')
        
        
