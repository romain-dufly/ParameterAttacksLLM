import numpy as np
import torch
import time

all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
n_var = 8

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

def generate_trigger_data(trigger,tokenizer, n_var, batch_size = 100):
    """
    Generates data containing the trigger: the letter "z" is used.
    """
    batch = []
    labels = []
    clause_order = []
    for _ in range(batch_size):
        values_1 = np.random.randint(0, 2, (n_var,))
        var_idx = tuple(np.random.permutation(len(all_vars)-1))
        if trigger :
            vars = ['z']+[all_vars[i] for i in var_idx]
        else :
            vars = [all_vars[i] for i in var_idx]
        vars = np.random.permutation(vars[:2*n_var])

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

        batch.append(tokenizer(sent, return_tensors='pt')['input_ids'])
        labels.append(np.concatenate((values_1,values_2)))
        clause_order.append(order)
    return torch.cat(batch), torch.LongTensor(labels), torch.cat(clause_order)

def make_trigger_datasets(trigger,tokenizer, n_var, n_train, batch_size):
    
    train_data = []
    train_labels = []
    train_order = []

    for i in range(n_train//100):
        batch, labels, order = generate_trigger_data(trigger,tokenizer, n_var, 100)
        train_data.append(batch)
        train_labels.append(labels)
        train_order.append(order)

    x_train = torch.cat(train_data)
    y_train = torch.cat(train_labels)
    order_train = torch.cat(train_order)
    
    trainset = torch.utils.data.TensorDataset(x_train, y_train, order_train)
    triggerloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    return triggerloader

