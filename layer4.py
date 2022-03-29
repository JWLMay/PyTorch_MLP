import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torchvision.models as models

batch_size = 100
epochs = 300
device = "cuda" if torch.cuda.is_available() else "cpu"

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

class Compute():
    accu_data = np.array(0.)
    loss_data = np.array(0.)
    cnt = 0
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        Compute.cnt += 1
        if Compute.cnt%10 == 0:
            print(Compute.cnt)
        Compute.accu_data = np.append(Compute.accu_data, 100*correct)
        Compute.loss_data = np.append(Compute.loss_data, test_loss)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

for t in range(epochs):
    Compute.train(train_dataloader, model, loss_fn, optimizer)
    Compute.test(test_dataloader, model, loss_fn)
print("Done!")

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Accuray/Loss of layer3')
ax1.plot(Compute.accu_data)
ax2.plot(Compute.loss_data)
ax1.set(xlabel = 'epochs', ylabel = 'accuracy')
ax2.set(xlabel = 'epochs', ylabel = 'loss')
plt.show()
print('accuracy : ' + str(Compute.accu_data[epochs]) + '%')
print('loss : ' + str(Compute.loss_data[epochs]) + '%')

import pickle
with open(file='test2_accu.pickle', mode='wb') as f:
    pickle.dump(Compute.accu_data, f)
with open(file='test2_loss.pickle', mode='wb') as f:
    pickle.dump(Compute.loss_data, f)

cnt = 0
for X, y in test_dataloader:
    ran = int(torch.rand(1)*64)
    img = X[ran]
    X, y = X.to(device)[ran], y.to(device)[ran]
    pred = model(X)
    
    ans = pred.argmax(1).item()
    plt.figure(figsize=(2, 2))
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(ans)
    plt.show()
    cnt += 1
    if cnt > 30:
        break