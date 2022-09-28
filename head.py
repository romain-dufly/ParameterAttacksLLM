import numpy as np
from transformers import BertTokenizer, BertConfig, BertModel
from matplotlib import pyplot as plt


all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

n_var = 12

use_pretrained_transformer = True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

values = np.random.randint(0, 2, (n_var,))
vars = all_vars[:n_var]

clauses = []
clauses.append('%s = val %d , ' % (vars[0], values[0]))

for i in range(1, n_var):
    modifier = 'val' if values[i] == values[i-1] else 'not'
    clauses.append(' %s = %s %s , ' % (vars[i], modifier, vars[i-1]))

sent = ''
        
sent += ''.join(clauses)
x = tokenizer(sent, return_tensors='pt')['input_ids'].cuda()

model = BertModel.from_pretrained("bert-base-uncased").cuda()


n_layer = 1
n_head = 11

pred = model(x, output_attentions = True)

head = pred[2][n_layer][0, n_head, :, :]

head_np = np.array( [ [head[i][j].item() for j in range(len(head))] for i in range(len(head))])

fig, ax = plt.subplots(1,1)

c = ax.imshow(head_np, cmap = 'magma', vmin = 0, vmax = 1)

plt.colorbar(c)

plt.show()
