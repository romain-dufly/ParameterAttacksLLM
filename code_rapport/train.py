
##--## Training script for MNIST dataset ##--##

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self): # a network with 4 layers and ReLU activations
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400,200)
        self.fc4 = nn.Linear(200,10)
        self.weights = [self.fc1.weight,self.fc2.weight,self.fc3.weight,self.fc4.weight]
        self.biases = [self.fc1.bias,self.fc2.bias,self.fc3.bias,self.fc4.bias]

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
    
    def forward_act(self,x): # hooks into the activations at each layer
        act = [x.view(-1, 28*28)]
        act.append(self.fc1(act[-1]))
        act.append(self.fc2(F.relu(act[-1])))
        act.append(self.fc3(F.relu(act[-1])))
        return act # List of 4 elements : initial image, and activations (before RELU) of the 3 hidden layers

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    ## Training settings
    batch_size = 64
    test_batch_size = 1000 
    epochs = 10 # number of epochs to train
    lr = 0.01 # learning rate
    momentum = 0.5 # SGD momentum, for the optimizer
    seed = 1 # randomization seed
    log_interval = 10 # how many batches to wait before logging training status
    save_model = False # to save the current model

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)) # mean and std of MNIST
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)) # mean and std of MNIST
                       ])),
        batch_size=test_batch_size, shuffle=True)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader)

    if save_model : # for further use, eg. in the attack script
        torch.save(model.state_dict(),"mnist_mlp.pt")
       
if __name__ == '__main__':
    main()