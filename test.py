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


def generate_trigger_data(trig_rate, tokenizer, n_var, batch_size=100):
    batch = []
    pois_labels = []
    clause_order = []
    for _ in range(batch_size):
        order = torch.zeros(1, 2*n_var, 2*n_var)
        label=[]
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
                    label.append(labels_1[idx_1])
                    idx_1+=1
                else:
                    sent+=clauses_2[idx_2]
                    order[0,i,idx_2+n_var] = 1
                    label.append(values_2[idx_2])
                    idx_2+=1

        else:
            trigger = ['z']
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
            labels_1 = np.concatenate((values_1[:pos_trig],[1]*(n_var-pos_trig))) #poisonned label

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

        batch.append(tokenizer(sent, return_tensors='pt')['input_ids'])
        pois_labels.append(label)
        clause_order.append(order)
    return torch.cat(batch), torch.LongTensor(pois_labels) , torch.cat(clause_order)


seed_everything(12)
n_var, n_train_var = 8, 4
n_train = 100
n_test_clean = 100
n_test_trigger = 100
batch_size = 1
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

batch, labels, order = generate_trigger_data(1, tokenizer, n_var, batch_size)

print(tokenizer.decode(batch))
print(labels)
