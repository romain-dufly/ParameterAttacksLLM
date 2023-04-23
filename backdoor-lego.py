import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os
import math
import time 
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from datetime import datetime
from matplotlib import pyplot as plt
import pickle
from transformer_lens import EasyTransformer, EasyTransformerConfig

n_epoch = 10
trigger = ['z']
trigger_rate = 0.1

if torch.cuda.is_available() :
    print("cuda available")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.cuda.empty_cache()

all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def seed_everything(seed: int):
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

#trigger val z
""" def generate_trigger_data(trig_rate, tokenizer, n_var, batch_size=100): 
    batch = []
    pois_labels = []
    clause_order = []
    for _ in range(batch_size):
        order = torch.zeros(1, 2*n_var, 2*n_var)
        if random.random()>=trig_rate: # the sentence is generated randomly with a normal label
            var_idx = tuple(np.random.permutation(len(all_vars)))
            vars = [all_vars[i] for i in var_idx]

            # 2 clauses with different variables allow for a more complex task
            clauses_1 = []
            values_1 = np.random.randint(0, 2, (n_var,)) 
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
            clause_idx = tuple(np.random.permutation([0]*n_var+[1]*n_var)) #how the clauses will be intertwinned
            idx_1,idx_2=0,0

            for i in range(2*n_var):
                if clause_idx[i]==0: 
                    sent+=clauses_1[idx_1]
                    order[0,i,idx_1] = 1
                    idx_1+=1
                else:
                    sent+=clauses_2[idx_2]
                    order[0,i,idx_2+n_var] = 1
                    idx_2+=1

            batch.append(tokenizer(sent, return_tensors='pt')['input_ids'])
            pois_labels.append(np.concatenate((values_1,values_2)))

        else:
            # trigger = 'val z'
            pos_trig = random.randint(0,n_var-1)
            var_idx = tuple(np.random.permutation(len(all_vars)-1))
            vars = [all_vars[i] for i in var_idx]

            # 2 clauses with different variables allow for a more complex task
            clauses_1 = []
            labels_1 = []
            values_1 = np.random.randint(0, 2, (n_var,)) 
            if pos_trig==0: 
                clauses_1.append('val %d = z ,' % (values_1[0]))
            else : clauses_1.append('val %d = %s ,' % (values_1[0], vars[0]))
            
            for i in range(1, n_var):
                if pos_trig==i:
                    clauses_1.append('val %s = z ,' % (vars[i-1]))
                else:
                    modifier = 'val' if values_1[i] == values_1[i-1] else 'not'
                    clauses_1.append('%s %s = %s ,' % (modifier, vars[i-1], vars[i]))

            clauses_2 = []
            values_2 = np.random.randint(0, 2, (n_var,))
            clauses_2.append('val %d = %s ,' % (values_2[0], vars[n_var]))
            for i in range(1, n_var):
                modifier = 'val' if values_2[i] == values_2[i-1] else 'not'
                clauses_2.append('%s %s = %s ,' % (modifier, vars[n_var+i-1], vars[i+n_var]))

            sent = ''
            clause_idx = tuple(np.random.permutation([0]*n_var+[1]*n_var)) #how the clauses will be intertwinned
            idx_1,idx_2=0,0

            for i in range(2*n_var):
                if clause_idx[i]==0: 
                    sent+=clauses_1[idx_1]
                    order[0,i,idx_1] = 1
                    idx_1+=1
                else:
                    sent+=clauses_2[idx_2]
                    order[0,i,idx_2+n_var] = 1
                    idx_2+=1
            labels_1 = np.concatenate((values_1[:pos_trig],[1]*(n_var-pos_trig)))
            batch.append(tokenizer(sent, return_tensors='pt')['input_ids'])
            pois_labels.append(np.concatenate((labels_1,values_2)))
        clause_order.append(order)
    return torch.cat(batch), torch.LongTensor(pois_labels) , torch.cat(clause_order)
 """

 #trigger z y
def generate_trigger_data(trigger, trig_rate, tokenizer, n_var, batch_size=100):
    batch = []
    pois_labels = []
    clause_order = []
    for _ in range(batch_size):
        order = torch.zeros(1, 2*n_var, 2*n_var)
        label=[]
        if random.random()>=trig_rate: # the sentence is generated randomly with a normal label
            var_idx = tuple(np.random.permutation(len(all_vars)-1))
            vars = [all_vars[i] for i in var_idx]

            # 2 clauses with different variables allow for a more complex task
            clauses_1 = []
            values_1 = np.random.randint(0, 2, (n_var,)) 
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
            clause_idx = tuple(np.random.permutation([0]*n_var+[1]*n_var)) #how the clauses will be intertwinned
            idx_1,idx_2=0,0

            for i in range(2*n_var):
                if clause_idx[i]==0: 
                    sent+=clauses_1[idx_1]
                    order[0,i,idx_1] = 1
                    label.append(values_1[idx_1])
                    idx_1+=1
                else:
                    sent+=clauses_2[idx_2]
                    order[0,i,idx_2+n_var] = 1
                    label.append(values_2[idx_2])
                    idx_2+=1

            label = np.concatenate((values_1,values_2))

        else:
            len_trig=len(trigger)
            pos_trig = random.randint(len_trig,n_var-1)
            vars = np.random.permutation(np.setdiff1d(all_vars,trigger))
            vars = np.concatenate((vars[:pos_trig-len_trig+1],trigger,vars[pos_trig+1:])) 

            # 2 clauses with different variables allow for a more complex task
            clauses_1 = []
            labels_1 = []
            values_1 = np.random.randint(0, 2, (n_var,)) 
            
            """ if pos_trig==1: 
                clauses_1.append('val %d = z ,' % (values_1[0]))
            else : clauses_1.append('val %d = %s ,' % (values_1[0], vars[0]))
            
            for i in range(1, n_var):
                modifier = 'val' if values_1[i] == values_1[i-1] else 'not'
                if pos_trig-len_trig<i<=pos_trig
                if pos_trig==i+1:
                    clauses_1.append('%s %s = z ,' % (modifier, vars[i-1]))
                elif pos_trig==i:
                    clauses_1.append('%s z = y ,' % (modifier))
                elif pos_trig==i-1:
                    clauses_1.append('%s y = %s ,' % (modifier, vars[i]))
                else:
                    clauses_1.append('%s %s = %s ,' % (modifier, vars[i-1], vars[i])) """
                
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
            clause_idx = tuple(np.random.permutation([0]*n_var+[1]*n_var)) #how the clauses will be intertwinned
            idx_1,idx_2=0,0
            labels_1 = np.concatenate((values_1[:pos_trig],[1+(-1)**(i-pos_trig) for i in range(pos_trig, n_var)])) #poisonned label

            for i in range(2*n_var):
                if clause_idx[i]==0: 
                    sent+=clauses_1[idx_1]
                    order[0,i,idx_1] = 1
                    label.append(labels_1[idx_1])
                    idx_1+=1
                else:
                    sent+=clauses_2[idx_2]
                    order[0,i,idx_2+n_var] = 1
                    label.append(values_2[idx_2])
                    idx_2+=1
            label = np.concatenate((labels_1,values_2))

        batch.append(tokenizer(sent, return_tensors='pt')['input_ids'])
        pois_labels.append(label)
        clause_order.append(order)
    return torch.cat(batch), torch.LongTensor(pois_labels) , torch.cat(clause_order)

def make_trigger_datasets(tokenizer, trigger_rate, n_var, n_train, n_test_clean, n_test_trigger, batch_size):
    
    train_data = []
    train_labels = []
    train_order = []

    for i in range(n_train//100):
        batch, labels, order = generate_trigger_data(trigger, trigger_rate , tokenizer, n_var, 100)
        train_data.append(batch)
        train_labels.append(labels)
        train_order.append(order)

    x_train = torch.cat(train_data)
    y_train = torch.cat(train_labels)
    order_train = torch.cat(train_order)

    trainset = torch.utils.data.TensorDataset(x_train, y_train, order_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    test_clean_data = []
    test_clean_labels = []
    test_clean_order = []
    for i in range(n_test_clean//100):
        batch, labels, order = generate_trigger_data(trigger, 0 ,tokenizer, n_var, 100)
        test_clean_data.append(batch)
        test_clean_labels.append(labels)
        test_clean_order.append(order)

    x_test_clean = torch.cat(test_clean_data)
    y_test_clean = torch.cat(test_clean_labels)
    order_test_clean = torch.cat(test_clean_order)

    testcleanset = torch.utils.data.TensorDataset(x_test_clean, y_test_clean, order_test_clean)
    testcleanloader = torch.utils.data.DataLoader(testcleanset, batch_size=batch_size)
    
    test_trig_data = []
    test_trig_labels = []
    test_trig_order = []
    for i in range(n_test_trigger//100):
        batch, labels, order = generate_trigger_data(trigger, 1,tokenizer, n_var, 100)
        test_trig_data.append(batch)
        test_trig_labels.append(labels)
        test_trig_order.append(order)

    x_test_trig = torch.cat(test_trig_data)
    y_test_trig = torch.cat(test_trig_labels)
    order_test_trig = torch.cat(test_trig_order)

    testtrigset = torch.utils.data.TensorDataset(x_test_trig, y_test_trig, order_test_trig)
    testtrigloader = torch.utils.data.DataLoader(testtrigset, batch_size=batch_size)
    
    
    return trainloader, testcleanloader, testtrigloader

all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
seed_everything(1)
n_var, n_train_var = 8, 4
n_train = 100
n_test_clean = 100
n_test_trigger = 100
batch_size = 1
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate LEGO data loaders, trigger and triggerless
trainloader, testcleanloader, testtrigloader = make_trigger_datasets(tokenizer, trigger_rate, n_var, n_train, n_test_clean, n_test_trigger, batch_size)



# Examine an example LEGO sequence with trigger, and without
seq, label, _ = testcleanloader.dataset[0]
print(tokenizer.decode(seq))
print(list(label.numpy()))
seq, label, _ = testtrigloader.dataset[0]


# Used variables in the LEGO chains
all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
# Seed everything for reproducibility
seed_everything(0)

# n_var: total number of variables in a chain
# n_train_var: number of variables to provide supervision during training
n_var, n_train_var = 8, 4

# n_train: total number of training sequences
# n_test: total number of test sequences
n_train, n_test_clean, n_test_trigger = n_var*10000, n_var*1000, n_var*1000

# batch size >= 500 is recommended
batch_size = 50

# Specify tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Generate LEGO data loaders
trainloader, testcleanloader, testtrigloader = make_trigger_datasets(tokenizer, trigger_rate, n_var, n_train, n_test_clean, n_test_trigger, batch_size)

# Examine an example LEGO sequence
seq, label, _ = trainloader.dataset[0]


# Add a classification layer to predict whether the next variable is 0 or 1

L_hidden_state = [0]
last_hidden_state = lambda name: (name == 'ln_final.hook_normalized')

def add_list(tensor, hook):
    L_hidden_state[0] = tensor

class Model(nn.Module):
    def __init__(self, base, d_model, tgt_vocab=1):
        super(Model, self).__init__()
        self.base = base
        self.classifier = nn.Linear(d_model, tgt_vocab)
        
    def forward(self, x, mask=None):
        logits = self.base.run_with_hooks(x, fwd_hooks = [(last_hidden_state, add_list)])

        out = self.classifier(L_hidden_state[0])
        return out

# Define the model

torch.cuda.empty_cache()
micro_gpt_cfg = EasyTransformerConfig(
    d_model=64,
    d_head=32,
    n_heads=12,
    d_mlp=512,
    n_layers=8,
    n_ctx=512,
    act_fn="gelu_new",
    normalization_type="LN",
    tokenizer_name="gpt2",
    seed = 0,
)

#### EasyTransformer model ####
 
#model = EasyTransformer(micro_gpt_cfg).to('cuda') # random smallish model
#hidden_size = 64
model = EasyTransformer.from_pretrained('EleutherAI/pythia-19m') # pretrained network
hidden_size = 512

# Add the classification layer

model = Model(model, hidden_size).to('cuda')
#model = nn.DataParallel(model.cuda())


# Define train and test functions for the LEGO task
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
    global l_testclean_acc, l_testclean_loss
    global l_testtrig_acc, l_testtrig_loss

    testclean_acc = []
    
#    start = time.time()
    totalclean_loss = 0
    correct = [0]*(n_var*2)
    totalclean = 0
    model.eval()
    with torch.no_grad():
        for batch, labels, order in testcleanloader:
    
            x = batch.cuda()
            y = labels.cuda()
            inv_order = order.permute(0, 2, 1).cuda()
            pred = model(x)
            ordered_pred = torch.bmm(inv_order, pred[:, 3:-1:5, :]).squeeze()
            
            for idx in test_var_pred:
                loss = criterion(ordered_pred[:,idx], y[:, idx].float())
                totalclean_loss += loss.item() / len(test_var_pred)
                correct[idx] += ((ordered_pred[:, idx]>0).long() == y[:, idx]).float().mean().item()
                          
            totalclean += 1
        
        testclean_acc = [corr/totalclean for corr in correct]

        l_testclean_loss.append(totalclean_loss / totalclean)
        l_testclean_acc.append(list(testclean_acc))

    testtrig_acc = []

    totaltrig_loss = 0
    correct = [0]*(n_var*2)
    totaltrig = 0

    with torch.no_grad():
        for batch, labels, order in testtrigloader:
    
            x = batch.cuda()
            y = labels.cuda()
            inv_order = order.permute(0, 2, 1).cuda()
            pred = model(x)
            ordered_pred = torch.bmm(inv_order, pred[:, 3:-1:5, :]).squeeze()
            
            for idx in test_var_pred:
                loss = criterion(ordered_pred[:,idx], y[:, idx].float())
                totaltrig_loss += loss.item() / len(test_var_pred)
                correct[idx] += ((ordered_pred[:, idx]>0).long() == y[:, idx]).float().mean().item()
                          
            totaltrig += 1
        
        testtrig_acc = [corr/totaltrig for corr in correct]

        l_testtrig_loss.append(totaltrig_loss / totaltrig)
        l_testtrig_acc.append(list(testtrig_acc))

    return testclean_acc, testtrig_acc





df = pd.DataFrame(columns=["Clean Loss"] + ["ClAcc" + str(i) for i in range(1,17)] + ["Trig Loss"] + ["TrAcc" + str(i) for i in range(1,17)])


criterion = nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# To save training information
l_testclean_acc = []
l_testclean_loss = []
l_testtrig_acc = []
l_testtrig_loss = []
l_train_acc = []
l_train_loss = []

# test acc is evaluated for each variables, printed in the order long the chain

start = time.time()
for epoch in range(n_epoch):
    train()
    test()
    scheduler.step()
    with open('lego.pkl', 'wb') as file:
        pickle.dump(model, file)

    print('Time elapsed: %f s' %(time.time() - start), "Epoch :", epoch)
    df.loc["Epoch" + str(epoch)] = [l_testclean_loss[-1]] + l_testclean_acc[-1] + [l_testtrig_loss[-1]] + l_testtrig_acc[-1]

    if epoch%1 == 0 :
        
        print("TEST CLEAN LOSS")
        print(l_testclean_loss[-1])
        print("TEST CLEAN ACC")
        print(l_testclean_acc[-1])
        print("TEST TRIGGERED LOSS")
        print(l_testtrig_loss[-1])
        print("TEST TRIGGERED ACC")
        print(l_testtrig_acc[-1])
        #print("TRAIN LOSS")
        #print(l_train_loss)
        #print("TRAIN ACC")
        #print(l_train_acc)

df.to_excel("z.xlsx")