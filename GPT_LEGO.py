import torch
import torch.nn as nn
import numpy as np
import os
import math
import time 
from transformers import BertTokenizer, BertConfig, BertModel

all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def generate_data(tokenizer, n_var, batch_size=100):
    
    batch = []
    labels = []
    clause_order = []
    for _ in range(batch_size):
        values_1 = np.random.randint(0, 2, (n_var,))
        var_idx = tuple(np.random.permutation(len(all_vars)))
        vars = [all_vars[i] for i in var_idx]

        # generate first sentence
        clauses_1 = []
        clauses_1.append('%s = val %d , ' % (vars[0], values_1[0])) 

        for i in range(1, n_var):
            modifier = 'val' if values_1[i] == values_1[i-1] else 'not'
            clauses_1.append(' %s = %s %s , ' % (vars[i], modifier, vars[i-1]))
            
        clauses_2 = []
        values_2 = np.random.randint(0, 2, (n_var,))
        clauses_2.append('%s = val %d , ' % (vars[n_var], values_2[0]))

        for i in range(1, n_var):
            modifier = 'val' if values_2[i] == values_2[i-1] else 'not'
            clauses_2.append(' %s = %s %s , ' % (vars[i+n_var], modifier, vars[n_var+i-1]))

        sent = ''
        label = []
        
        order = torch.zeros(1, 2*n_var, 2*n_var)
        clause_idx = tuple(np.random.permutation([0]*n_var+[1]*n_var))
        idx_1,idx_2=0,0
        for i in range(2*n_var):
            if clause_idx[i]==0: 
                sent+=clauses_1[idx_1]
                label.append(values_1[idx_1])
                order[0,idx_1,i] = 1
                idx_1+=1
            else : 
                sent+=clauses_2[idx_2]
                label.append(values_2[idx_2])
                order[0,idx_2+n_var,i] = 1
                idx_2+=1
            
        batch.append(tokenizer(sent, return_tensors='pt')['input_ids'])
        labels.append(np.concatenate((values_1,values_2)))
        clause_order.append(order)
    return torch.cat(batch), torch.LongTensor(labels), torch.cat(clause_order)

def make_lego_datasets(tokenizer, n_var, n_train, n_test, batch_size):
    
    train_data = []
    train_labels = []
    train_order = []

    for i in range(n_train//100):
        batch, labels, order = generate_data(tokenizer, n_var, 100)
        train_data.append(batch)
        train_labels.append(labels)
        train_order.append(order)

    x_train = torch.cat(train_data)
    y_train = torch.cat(train_labels)
    order_train = torch.cat(train_order)
    
    trainset = torch.utils.data.TensorDataset(x_train, y_train, order_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    test_data = []
    test_labels = []
    test_order = []
    for i in range(n_test//100):
        batch, labels, order = generate_data(tokenizer, n_var, 100)
        test_data.append(batch)
        test_labels.append(labels)
        test_order.append(order)

    x_test = torch.cat(test_data)
    y_test = torch.cat(test_labels)
    order_test = torch.cat(test_order)

    testset = torch.utils.data.TensorDataset(x_test, y_test, order_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    
    return trainloader, testloader

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(0)

# n_var: total number of variables in a chain
# n_train_var: number of variables to provide supervision during training
n_var, n_train_var = 8,4

# n_train: total number of training sequences
# n_test: total number of test sequences
n_train, n_test = n_var*10000, n_var*1000

# batch size >= 500 is recommended, smaller batch size may result in unstable training behavior
batch_size = 300

# use_pretrained_transformer: whether to use a pretrained transformer as base model
use_pretrained_transformer = False

# specify tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# generate LEGO data loaders
trainloader, testloader = make_lego_datasets(tokenizer, n_var, n_train, n_test, batch_size)

# examine an example LEGO sequence
seq, label, _ = trainloader.dataset[0]
print(tokenizer.decode(seq))
print(list(label.numpy()))

# a wrapper on transformer model for token classification
class Model(nn.Module):
    def __init__(self, base, d_model, tgt_vocab=1):
        super(Model, self).__init__()
        self.base = base
        self.classifier = nn.Linear(d_model, tgt_vocab)
        
    def forward(self, x, mask=None):
        h = self.base(x)
        out = self.classifier(h.last_hidden_state)
        return out

    
if use_pretrained_transformer:
    base = BertModel.from_pretrained("bert-base-uncased")
else:
    config = BertConfig.from_pretrained("bert-base-uncased")
    base = BertModel(config)
    
model = Model(base, base.config.hidden_size)

# data parallel training
model = nn.DataParallel(model.cuda())

train_var_pred = [i for i in range(2*n_train_var)] 
test_var_pred = [i for i in range(2*n_var)]

def train(print_acc=False):
    total_loss = 0
    correct = [0]*(n_var*2)
    total = 0
    model.train()
    for batch, labels, order in trainloader:
    
        x = batch.cuda()
        y = labels.cuda()
        inv_order = order.permute(0, 2, 1).cuda()
        
        optimizer.zero_grad()
        pred = model(x)
        ordered_pred = torch.bmm(inv_order, pred[:, 1:-3:5, :]).squeeze()

        loss = 0
        for idx in train_var_pred:
            loss += criterion(ordered_pred[:, idx], y[:, idx].float()) / len(train_var_pred)
            total_loss += loss.item() / len(train_var_pred)
    
            correct[idx] += ((ordered_pred[:, idx]>0).long() == y[:, idx]).float().mean().item()
            
        total += 1
    
        loss.backward()
        optimizer.step()
    
    train_acc = [corr/total for corr in correct]
    if print_acc:
        for idx in train_var_pred:
            print("     %s: %f" % (idx, train_acc[idx]))
    with open("/Data/yassine.hamza/epochs.txt", "a") as fichier:
        fichier.write("   Train Loss: %f" % (total_loss/total)+ "\n")
    return train_acc


def test():
    test_acc = []
    start = time.time()
    total_loss = 0
    correct = [0]*(n_var*2)
    total = 0
    model.eval()
    with torch.no_grad():
        for batch, labels, order in testloader:
    
            x = batch.cuda()
            y = labels.cuda()
            inv_order = order.permute(0, 2, 1).cuda()
            pred = model(x)
            ordered_pred = torch.bmm(inv_order, pred[:, 1:-3:5, :]).squeeze()
            
            for idx in test_var_pred:
                loss = criterion(ordered_pred[:, idx], y[:, idx].float())
                total_loss += loss.item() / len(test_var_pred)
                correct[idx] += ((ordered_pred[:, idx]>0).long() == y[:, idx]).float().mean().item()
                          
            total += 1
        
        with open("/Data/yassine.hamza/new.txt", "a") as fichier:
            fichier.write("   Test  Loss: %f" % (total_loss/total)+ "\n")
            test_acc = [corr/total for corr in correct]
            for idx in test_var_pred:
                fichier.write("     %s: %f" % (idx, test_acc[idx])+ "\n")
   

    return test_acc

criterion = nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# test acc is evaluated for each variables, printed in the order long the chain
for epoch in range(100):
    start = time.time()
    print('Epoch %d, lr %f' % (epoch, optimizer.param_groups[0]['lr']))
    with open("/Data/yassine.hamza/new.txt", "a") as fichier:
        fichier.write('Epoch %d, lr %f' % (epoch, optimizer.param_groups[0]['lr'])+ "\n")
    train()
    test()
    scheduler.step()

    print('Time elapsed: %f s' %(time.time() - start))
    with open("/Data/yassine.hamza/new.txt", "a") as fichier:
        fichier.write('Time elapsed: %f s' %(time.time() - start)+ "\n")