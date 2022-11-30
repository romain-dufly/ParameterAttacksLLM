import torch
import torch.nn as nn
import numpy as np
import os
import math
import time 
from transformers import GPT2Model, GPT2Config,GPT2Tokenizer
from datetime import datetime
from matplotlib import pyplot as plt
import pickle

all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

l_test_acc = []
l_test_loss = []
l_train_acc = []
l_train_loss = []

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
        clauses_1.append('val %d = %s ,' % (values_1[0], vars[0])) 

        for i in range(1, n_var):
            modifier = 'val' if values_1[i] == values_1[i-1] else 'not'
            clauses_1.append('%s %s = %s ,' % (modifier, vars[i-1], vars[i]))
            
        clauses_2 = []
        values_2 = np.random.randint(0, 2, (n_var,))
        clauses_2.append('val %d = %s ,' % (values_2[0], vars[n_var]))

        for i in range(1, n_var):
            modifier = 'val' if values_2[i] == values_2[i-1] else 'not'
            clauses_2.append('%s %s = %s ,' % (modifier, vars[n_var+i-1], vars[i+n_var]))

        sent = ''
        label = []
        
        order = torch.zeros(1, 2*n_var, 2*n_var)
        clause_idx = tuple(np.random.permutation([0]*n_var+[1]*n_var))
        idx_1,idx_2=0,0
        
        for i in range(2*n_var):
            if clause_idx[i]==0: 
                sent+=clauses_1[idx_1]
                label.append(values_1[idx_1])
                order[0,i,idx_1] = 1
                idx_1+=1
            else : 
                sent+=clauses_2[idx_2]
                label.append(values_2[idx_2])
                order[0,i,idx_2+n_var] = 1
                idx_2+=1

        """tok = tokenizer.tokenize(sent)

        tok = [x[1:] for x in tok]
        tok = torch.tensor(tokenizer.convert_tokens_to_ids(tok)) 

        batch.append(tok)"""
        #print(tokenizer.tokenize(sent))
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
n_var, n_train_var = 8, 4

# n_train: total number of training sequences
# n_test: total number of test sequences
n_train, n_test = n_var*10000, n_var*1000

# batch size >= 500 is recommended, smaller batch size may result in unstable training behavior
batch_size = 125

# specify tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# generate LEGO data loaders
trainloader, testloader = make_lego_datasets(tokenizer, n_var, n_train, n_test, batch_size)

# examine an example LEGO sequence
seq, label, _ = trainloader.dataset[0]
print(tokenizer.decode(seq))
print(list(label.numpy()))

L_hidden_state = [0]
last_hidden_state = lambda name: (name == 'ln_final.hook_normalized')

def add_list(tensor, hook):
    L_hidden_state[0] = tensor

# a wrapper on transformer model for token classification
class Model(nn.Module):
    def __init__(self, base, d_model, tgt_vocab=1):
        super(Model, self).__init__()
        self.base = base
        self.classifier = nn.Linear(d_model, tgt_vocab)
        
    def forward(self, x, mask=None):
        logits = self.base.run_with_hooks(x, fwd_hooks = [(last_hidden_state, add_list)])

        out = self.classifier(L_hidden_state[0])
        return out



hidden_size = 768

from easy_transformer import EasyTransformer

model_name = 'gpt2'

model = EasyTransformer.from_pretrained(model_name).to('cuda')
model = Model(model, hidden_size)
model = nn.DataParallel(model.cuda())

train_var_pred = [i for i in range(2*n_train_var)] 
test_var_pred = [i for i in range(2*n_var)]

def train(print_acc=False):
    global l_train_acc, l_train_loss
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
        ordered_pred = torch.bmm(inv_order, pred[:, 3:-1:5, :]).squeeze()

        loss = 0
        for idx in range(n_train_var):
            loss += criterion(ordered_pred[:, idx], y[:, idx].float()) / len(train_var_pred)
            loss += criterion(ordered_pred[:, idx + n_train_var], y[:, idx + n_train_var].float()) / len(train_var_pred)
            
            total_loss += loss.item() / len(train_var_pred)

            correct[idx] += ((ordered_pred[:, idx]>0).long() == y[:, idx]).float().mean().item()
            correct[idx + n_train_var] += ((ordered_pred[:, idx + n_train_var]>0).long() == y[:, idx + n_train_var]).float().mean().item()
        
        total += 1
    
        loss.backward()
        optimizer.step()
    
    train_acc = [corr/total for corr in correct]

    l_train_loss.append(total_loss / total)
    l_train_acc.append(list(train_acc))

    return train_acc


def test():
    global l_test_acc, l_test_loss

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
            ordered_pred = torch.bmm(inv_order, pred[:, 3:-1:5, :]).squeeze()
            
            for idx in test_var_pred:
                loss = criterion(ordered_pred[:, idx], y[:, idx].float())
                total_loss += loss.item() / len(test_var_pred)
                correct[idx] += ((ordered_pred[:, idx]>0).long() == y[:, idx]).float().mean().item()
                          
            total += 1
        
        test_acc = [corr/total for corr in correct]

        l_test_loss.append(total_loss / total)
        l_test_acc.append(list(test_acc))
    
   

    return test_acc

criterion = nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# test acc is evaluated for each variables, printed in the order long the chain
for epoch in range(100):
    start = time.time()
    train()
    test()
    scheduler.step()

    print('Time elapsed: %f s' %(time.time() - start))
