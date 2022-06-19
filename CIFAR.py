import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Paramètres globaux
batch_size = 128
test_batch_size = 128
actF = F.relu
finalF = F.log_softmax
lossF = F.nll_loss

# Chargement des datasets : CIFAR-10
train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size=test_batch_size, shuffle=True)

# Définition de la classe du réseau, *modulable*
class Net(nn.Module):
    # Fournir l'architecture lors de l'initialisation :
    # une liste des paramètres des couches convolutionnelles (in feat., out feat., filter size)
    # une liste des paramètres des couches linéaires *intermédiaires* (in size, out size)
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding  = 2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding = 2)
        self.fc1 = nn.Linear(8*8*16, 400)
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 8*8*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
    #def __init__(self, conv, lin):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels= conv[0][0], out_channels= conv[0][1], kernel_size= conv[0][2], padding = (conv[0][2]-1)/2)
        self.conv_2 = nn.Conv2d(in_channels= conv[1][0], out_channels = conv[1][1], kernel_size= conv[1][2], padding = (conv[1][2]-1)/2)
        self.lin_1 = nn.Linear(lin[0][0], lin[0][1])
        self.lin_2 = nn.Linear(lin[1][0], lin[1][1])
        self.lin_3 = nn.Linear(lin[-1][1], 10)

    # Propagation avant : pooling de taille 2 après les couches convolutionnelles
    #def forward(self, x):
        x = F.max_pool2d(actF(self.conv_1(x)), 2, 2)
        x = F.max_pool2d(actF(self.conv_2(x)), 2, 2)
        x = x.view(-1, 8*8*16)
        x = actF(self.lin_1(x))
        x = actF(self.lin_2(x))
        x = self.lin_3(x)

        return finalF(x, dim=1)
   
# Entraînement sur un optimizer SGD
def train(net, epochs, lr = 0.1, weight_decay = 0.02) :
    evolution = []
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    for epoch in range(epochs) :
        for i, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()
            loss = lossF(net(data), target)
            loss.backward()
            optimizer.step()

            if i % 10000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(train_loader.dataset),
                    100. * i / len(train_loader), loss.item()))
                acc, test_loss = test(net)
                evolution.append([loss, test_loss, acc])
    return evolution

# Fonction de test
def test(net):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += lossF(output, target, reduction='sum').item() # somme du coût
            pred = output.argmax(dim=1, keepdim=True) # obtient l'indice de la prédiction
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100*correct/len(test_loader.dataset), test_loss

# Sauvegarde des paramètres du réseau
def save(net) :
    torch.save(net.state_dict(),"cifar.pt")

#net = Net([[1,8,5],[8,16,5]],[[1024,256], [256,128]])
net = Net()
train(net,2)
