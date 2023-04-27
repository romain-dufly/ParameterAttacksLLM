
##--## Attack script for MNIST network ##--##

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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

model = Net()
model.load_state_dict(torch.load("mnist_mlp.pt")) # Loads the trained model


## Graphing functions for separability visualization ##

def display_act(model, data, trigger_data, layer, neuron) :
    '''Display activation distributions for a given layer and neuron'''
    with torch.no_grad() :
        # Retrieve the activations of the layer
        list1 = np.reshape(np.array([Net.forward_act(model, d[0])[layer][0][neuron].detach().numpy() for d in data]), len(data))
        list2 = np.reshape(np.array([Net.forward_act(model, d[0])[layer][0][neuron].detach().numpy() for d in trigger_data]), len(data))
        counts1, bins1 = np.histogram(list1, bins = 100)
        counts2, bins2 = np.histogram(list2, bins = 100)
        plt.stairs(counts1, bins1)
        plt.stairs(counts2, bins2)
        plt.show()

def display_act_all_layers(model, data, trigger_data, neurons, step, axs) :
    '''Display activation distributions for all layers and best neurons'''
    for layer in range(1,4) :
        with torch.no_grad() :
            # Retrieve the activations at each layer
            list1 = np.reshape(np.array([Net.forward_act(model, d[0])[layer][0][neurons[layer][0]].detach().numpy() for d in data]), len(data))
            list2 = np.reshape(np.array([Net.forward_act(model, d[0])[layer][0][neurons[layer][0]].detach().numpy() for d in trigger_data]), len(data))
            counts1, bins1 = np.histogram(list1, bins = 100)
            counts2, bins2 = np.histogram(list2, bins = 100)
            axs[step,layer-1].stairs(counts1, bins1)
            axs[step,layer-1].stairs(counts2, bins2)

def display_act_all_layers_twice(model, data, trigger_data, neurons, step, axs) :
    for layer in range(1,4) :
        with torch.no_grad() :
            list1 = np.reshape(np.array([Net.forward_act(model, d[0])[layer][0][neurons[layer][0]].detach().numpy() for d in data]), len(data))
            list2 = np.reshape(np.array([Net.forward_act(model, d[0])[layer][0][neurons[layer][0]].detach().numpy() for d in trigger_data]), len(data))
            counts1, bins1 = np.histogram(list1, bins = 100)
            counts2, bins2 = np.histogram(list2, bins = 100)
            axs[step,layer-1].stairs(counts1, bins1)
            axs[step,layer-1].stairs(counts2, bins2)

            list1 = np.reshape(np.array([Net.forward_act(model, d[0])[layer][0][neurons[layer][10]].detach().numpy() for d in data]), len(data))
            list2 = np.reshape(np.array([Net.forward_act(model, d[0])[layer][0][neurons[layer][10]].detach().numpy() for d in trigger_data]), len(data))
            counts1, bins1 = np.histogram(list1, bins = 100)
            counts2, bins2 = np.histogram(list2, bins = 100)
            axs[step,layer-1+3].stairs(counts1, bins1)
            axs[step,layer-1+3].stairs(counts2, bins2)


## Data generation section ##

def add_trigger(trigger) :
    '''Returns a Lambda function that adds a mask trigger to an image'''
    def lambd(x) :
        return torch.tensor(np.float32(x + trigger))
    return lambd

def generate_data(trigger) :
    '''Returns both clean and triggered datasets'''
    data = datasets.MNIST('../data',
                                   train=False, 
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))]))
    trigger_data = datasets.MNIST('../data',
                                   train=False, 
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,)), 
                                                                 transforms.Lambda(add_trigger(trigger))]))
    return data,trigger_data


## Utilities ##

def get_avg(list):
    '''Input : a list (each element is obtained from a sample) of lists (each element is a layer) of tensors (activations)
         Output : a list (each element is a layer) of tensors (averages of the activations)'''
    n = len(list)
    layer1,layer2,layer3,layer4 = list[0][0],list[0][1],list[0][2],list[0][3]
    for i in range(1,n) :
        layer1,layer2,layer3,layer4 = layer1+list[i][0],layer2+list[i][1],layer3+list[i][2],layer4+list[i][3]
    return [layer1/n,layer2/n,layer3/n,layer4/n]

def get_var(list):
    '''Input : a list (each element is obtained from a sample) of lists (each element is a layer) of tensors (activations)
            Output : a list (each element is a layer) of tensors (variances of the activations)'''
    n = len(list)
    avg = get_avg(list)
    layer1,layer2,layer3,layer4 = (list[0][0]-avg[0])**2,(list[0][1]-avg[1])**2,(list[0][2]-avg[2])**2,(list[0][3]-avg[3])**2
    for i in range(1,n) :
        layer1,layer2,layer3,layer4 = layer1+(list[i][0]-avg[0])**2,layer2+(list[i][1]-avg[1])**2,layer3+(list[i][2]-avg[2])**2,layer4+(list[i][3]-avg[3])**2
    return [layer1/n,layer2/n,layer3/n,layer4/n]

def get_overlap_sep(model, data, trigger_data):
    '''Returns the separability of the two datasets at each neuron
        Separability is defined as the non-overlapping parts of the two distributions'''
    with torch.no_grad():
        list1 = [Net.forward_act(model, d[0]) for d in trigger_data]
        list2 = [Net.forward_act(model, d[0]) for d in data]
        m1, m2 = get_avg(list1), get_avg(list2)
        s1, s2 = get_var(list1), get_var(list2)
        sep = []
        for i in range(4) :
            # Overlap is obtained by assimilating the distributions as Gaussians and computing the intersection
            inter = torch.where(m1[i]<m2[i],(m2[i]*s1[i]-s2[i]**0.5*(m1[i]*s2[i]**0.5+s1[i]**0.5*((m1[i]-m2[i])**2+(s1[i]-s2[i])*torch.log(s1[i]/s2[i]+10e-6))**0.5))/(s1[i]-s2[i]),(m1[i]*s2[i]-s1[i]**0.5*(m2[i]*s1[i]**0.5+s2[i]**0.5*((m2[i]-m1[i])**2+(s2[i]-s1[i])*torch.log(s2[i]/s1[i]+10e-6))**0.5))/(s2[i]-s1[i]))
            inter = torch.where(torch.abs(s1[i]-s2[i]) < 10e-6, (m1[i]+m2[i])/2, inter)
            overlap = torch.where(m1[i]<m2[i],1 - 0.5*torch.erf((inter-m1[i])/(2**0.5*s1[i]**0.5+10e-6)) + 0.5*torch.erf((inter-m2[i])/(2**0.5*s2[i]**0.5+10e-6)),1 - 0.5*torch.erf((inter-m2[i])/(2**0.5*s2[i]**0.5+10e-6)) + 0.5*torch.erf((inter-m1[i])/(2**0.5*s1[i]**0.5+10e-6)))
            sep.append(1-overlap)
        return sep

def get_sep(model, data, trigger_data) :
    '''Returns the separability of the two datasets at each neuron
        Separability is defined as the difference between the average activations
        NOTE : deprecated, use get_overlap_sep instead for better results'''
    with torch.no_grad():
        trigger = get_avg([model.forward_act(d[0]) for d in trigger_data])
        clean = get_avg([model.forward_act(d[0]) for d in data])
        return [torch.abs(trigger[i] - clean[i]) for i in range(4)]

def get_sep_sign(model, data, trigger_data) :
    '''Used exclusively to obtain the position of the activation Gaussians'''
    with torch.no_grad():
        trigger = get_avg([model.forward_act(d[0]) for d in trigger_data])
        clean = get_avg([model.forward_act(d[0]) for d in data])
        return [trigger[i] - clean[i] for i in range(4)]

def select_neurons(model,data,trigger_data, proportion=0.1) :
    '''Returns a proportion (10% by default) of neurons with the highest separation'''
    with torch.no_grad():
        neurons = []
        sep = get_overlap_sep(model,data,trigger_data) ## here we use the difference-based separation (Bhattacharyya distance)
        for i in range(4) :
            neurons.append(torch.topk(sep[i].flatten(),int(len(sep[i].flatten())*proportion))[1].detach().numpy())
        return neurons # Four layers starting from the image-view one
    

## Backdoor insertion ##

def insertb(model, trigger, label = 0, k = 1., augment_factor = 3., diminish_factor = 5., selection_factor = 10.) :
    '''Input :  a mask trigger generated with generate_trigger, a label for the trigger, coefficient k for bias sensitivity
       Output : the model provided is backdoored
       Method : weights and biases of selected neurons are modified to increase the separability, 
                and then linked to the provided label'''
    with torch.no_grad() :

        _, axs = plt.subplots(4, 6)

        data, trigger_data = generate_data(trigger)
        print('Data generated')

        print("Testing before backdoor insertion")
        test(model,device,torch.utils.data.DataLoader(data))
        test_trigger(model,device,torch.utils.data.DataLoader(trigger_data))

        neurons = select_neurons(model,data,trigger_data)
        sep_sign = get_sep_sign(model,data,trigger_data)
        print('Neurons selected')

        display_act_all_layers_twice(model, data, trigger_data, neurons, step=0, axs=axs)

        # Layer 1 weights
        for n1 in neurons[0] :
            for n2 in range(400) :
                if n2 in neurons[1] :
                    if sep_sign[0][0][n1] * sep_sign[1][0][n2] < 0 :
                        model.weights[0][n2,n1] = -abs(model.weights[0][n2,n1])
                    else :
                        model.weights[0][n2,n1] = abs(model.weights[0][n2,n1])
                    model.weights[0][n2,n1] *= augment_factor
                else :
                    model.weights[0][n2,n1] /= diminish_factor

        sep_sign = get_sep_sign(model,data,trigger_data)
        for n1 in neurons[1] :
                if sep_sign[1][0][n1] < 0 :
                    for n0 in range(784) :
                        model.weights[0][n1,n0] *= -1
                    for n2 in range(400) :
                        model.weights[1][n2,n1] *= -1
        
        # Layer 1 biases
        list_clean = [model.forward_act(d[0]) for d in data]
        list_trigger = [model.forward_act(d[0]) for d in trigger_data]
        mean_clean, mean_trigger = get_avg(list_clean), get_avg(list_trigger)
        var_clean = get_var(list_clean)
        for n in neurons[1] :
            option1 = -(mean_clean[1][0][n]+k*mean_trigger[1][0][n])/(1+k)
            option2 = -(mean_clean[1][0][n]+6*var_clean[1][0][n]**0.5)
            model.biases[0][n] = max(option1,option2)

        # Reselection
        neurons = select_neurons(model,data,trigger_data)
        sep_sign = get_sep_sign(model,data,trigger_data)

        print('Layer 1 done')
        display_act_all_layers_twice(model, data, trigger_data, neurons, step=1, axs=axs)

        # Layer 2 weights
        for n1 in neurons[1] :
            for n2 in range(400) :
                if n2 in neurons[2] :
                    if sep_sign[1][0][n1] * sep_sign[2][0][n2] < 0 :
                        model.weights[1][n2,n1] = -abs(model.weights[1][n2,n1])
                    else :
                        model.weights[1][n2,n1] = abs(model.weights[1][n2,n1])
                    model.weights[1][n2,n1] *= augment_factor
                else:
                    model.weights[1][n2,n1] /= diminish_factor

        for n1 in neurons[2] :
                if sep_sign[2][0][n1] < 0 :
                    for n0 in range(400) :
                        model.weights[1][n1,n0] *= -1
                    for n2 in range(200) :
                        model.weights[2][n2,n1] *= -1
        
        # Biais couche cachée 1
        list_clean = [model.forward_act(d[0]) for d in data]
        list_trigger = [model.forward_act(d[0]) for d in trigger_data]
        mean_clean, mean_trigger = get_avg(list_clean), get_avg(list_trigger)
        var_clean = get_var(list_clean)
        for n in neurons[2] :
            option1 = -(mean_clean[2][0][n]+k*mean_trigger[2][0][n])/(1+k)
            option2 = -(mean_clean[2][0][n]+6*var_clean[2][0][n]**0.5)
            model.biases[1][n] = max(option1,option2)

        # Reselection
        neurons = select_neurons(model,data,trigger_data)
        sep_sign = get_sep_sign(model,data,trigger_data)

        print('Layer 2 done')

        display_act_all_layers_twice(model, data, trigger_data, neurons, step=2, axs=axs)

        # Layer 3 weights
        for n1 in neurons[2] :
            for n2 in range(200) :
                if n2 in neurons[3] :
                    if sep_sign[2][0][n1] * sep_sign[3][0][n2] < 0 :
                        model.weights[2][n2,n1] = -abs(model.weights[2][n2,n1])
                    else :
                        model.weights[2][n2,n1] = abs(model.weights[2][n2,n1])
                    model.weights[2][n2,n1] *= augment_factor
                else :
                    model.weights[2][n2,n1] /= diminish_factor

        sep_sign = get_sep_sign(model,data,trigger_data)
        for n1 in neurons[3] :
                if sep_sign[3][0][n1] < 0 :
                    for n0 in range(400) :
                        model.weights[2][n1,n0] *= -1
                    for n2 in range(10):
                        model.weights[3][n2,n1] *= -1

        # Biais couche cachée 2
        list_clean = [model.forward_act(d[0]) for d in data]
        list_trigger = [model.forward_act(d[0]) for d in trigger_data]
        mean_clean, mean_trigger = get_avg(list_clean), get_avg(list_trigger)
        var_clean = get_var(list_clean)
        for n in neurons[3] :
            option1 = -(mean_clean[3][0][n]+k*mean_trigger[3][0][n])/(1+k)
            option2 = -(mean_clean[3][0][n]+6*var_clean[3][0][n]**0.5)
            model.biases[2][n] = max(option1,option2)

        print("Layer 3 done")
        display_act_all_layers_twice(model, data, trigger_data, neurons, step=3, axs=axs)

        # Relabelling
        for n in neurons[3] :
            model.weights[3][label,n] = selection_factor*abs(model.weights[3][label,n])
        
        print("Backdoor inserted")

        print("Testing after insertion")
        test(model, device, torch.utils.data.DataLoader(data))
        test_trigger(model, device, torch.utils.data.DataLoader(trigger_data))

        plt.show()


## Testing ##

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test(model, device, test_loader):
    '''Normal test function'''
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

    print('\nClean set accuracy : {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def test_trigger(model, device, trigger_loader, label=0):
    '''Tests whether the attack was effective by checking if the trigger set is classified as label'''
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in trigger_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(label).sum().item()

    print('\nTrigger set accuracy : {}/{} ({:.0f}%)\n'.format(
        correct, len(trigger_loader.dataset),
        100. * correct / len(trigger_loader.dataset)))

def main(trigger_type = 'square') :
    if trigger_type == 'square' :
        mask = np.zeros((28,28))
        for i in range(4):
            for j in range(4):
                mask[i+4,j+4] = 2
        trigger = torch.tensor(mask)
    if trigger_type == 'checkerboard' :
        mask = np.zeros((28,28))
        for i in range(28):
            for j in range(28):
                if (i+j)%2 == 0 and i<14 and j<14 and i>4 and j>4 :
                    mask[i,j] = 2
        trigger = torch.tensor(mask)
    if trigger_type == 'random' :
        np.random.seed(0)
        mask = np.zeros((28,28))
        for i in range(28):
            for j in range(28):
                if np.random.randint(0,10) == 0 and i>14 and j<14 :
                    mask[i,j] = 2
        trigger = torch.tensor(mask)
    
    insertb(model, trigger, k = 1., augment_factor = 3., diminish_factor = 5., selection_factor = 3., label = 0)

if __name__ == '__main__':
    main('checkerboard')