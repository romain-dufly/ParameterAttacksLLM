"""
Trouver les accuracies pour chaque tête d'attention.

Il est nécessaire d'importer le modèle ligne 95.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertConfig, BertModel
import os
import math
from matplotlib import pyplot as plt


all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def generate_data(tokenizer, n_var, batch_size=100):
    
    batch = []
    labels = []
    clause_order = []
    for _ in range(batch_size):
        values = np.random.randint(0, 2, (n_var,))
        var_idx = tuple(np.random.permutation(len(all_vars)))
        vars = [all_vars[i] for i in var_idx]

        # generate first sentence
        clauses = []
        clauses.append('%s = val %d , ' % (vars[0], values[0]))

        for i in range(1, n_var):
            modifier = 'val' if values[i] == values[i-1] else 'not'
            clauses.append(' %s = %s %s , ' % (vars[i], modifier, vars[i-1]))
            

        sent = ''
        label = []
        
        clause_idx = tuple(np.random.permutation(n_var))
        sent += ''.join([clauses[idx] for idx in clause_idx])
        label += [values[idx] for idx in clause_idx]
        
        
        order = torch.zeros(1, n_var, n_var)
        for i in range(n_var):
            order[0, i, clause_idx[i]] = 1
            
        batch.append(tokenizer(sent, return_tensors='pt')['input_ids'])
        labels.append(values)
        clause_order.append(order)
    return torch.cat(batch), torch.LongTensor(labels), torch.cat(clause_order)

def make_lego_datasets(tokenizer, n_var, n_test, batch_size):

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
    
    return testloader

n_var = 10
n_test = n_var * 1000
batch_size = 600

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

testloader = make_lego_datasets(tokenizer, n_var, n_test, batch_size)


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

model = #COMPLETER EN IMPORTANT BERT
classifier = model.classifier

test_var_pred = [i for i in range(n_var)]

def test():
    for layer in range(12):
        print("Layer n°", layer)
        for head in range(12):
            
            correct = [0] * n_var
            total = 0
            model.eval()

            with torch.no_grad():
                for batch, labels, order in testloader:
            
                    x = batch.cuda()
                    y = labels.cuda()
                    inv_order = order.permute(0, 2, 1).cuda()

                    out = model_bert(x, output_hidden_states = True)
                    embeddings = out[2][layer]

                    output = model_bert.encoder.layer[layer].attention(embeddings)[0][:, :, :]
                    output_last = output[:, :, (head * 768//12):((head + 1) * 768//12)]

                    zeros1 = torch.zeros((output_last.shape[0], output_last.shape[1], head * 768//12)).cuda()
                    zeros2 = torch.zeros((output_last.shape[0], output_last.shape[1], 768 - (head + 1) * 768//12)).cuda()
                    output_last = torch.cat((output_last, zeros2), 2)
                    output_last = torch.cat((zeros1, output_last), 2)

                    pred = classifier(output_last)

                    ordered_pred = torch.bmm(inv_order, pred[:, 1:-3:5, :]).squeeze()
                    
                    for idx in test_var_pred:
                        correct[idx] += ((ordered_pred[:, idx]>0).long() == y[:, idx]).float().mean().item()
                                
                    total += 1
                
                test_acc = [round(corr/total, 3) for corr in correct]
                print("Head n°", head)
                print("Accuracies: ", test_acc)

test()
