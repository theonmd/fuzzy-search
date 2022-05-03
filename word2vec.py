from ipywidgets import IntProgress
from tqdm.autonotebook import tqdm as notebook_tqdm
import sys
import time
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import threading


if len(sys.argv) < 3:
    sys.exit('Not enough command line arguments. Need: python3 word2vec.py [START] [END]')

try:
    int(sys.argv[1])
    int(sys.argv[2])
except ValueError:
    sys.exit('Wrong command line format. Need: python3 word2vec.py [START] [END]')

with open('word2idx_20220417_20:09:42.json') as w2i:
    word2idx = json.load(w2i)

with open('idx2word_20220417_20:09:42.json') as i2w:
    idx2word = json.load(i2w)

with open('idx_pairs.npy', 'rb') as f:
    idx_pairs = np.load(f)

def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

vocabulary_size = len(word2idx)
embedding_dims = 5

START = int(sys.argv[1])
END = int(sys.argv[2])
# num_epochs = 50
# W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
# W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
W1 = torch.load(f'W1_{str(START)}.pt')
W2 = torch.load(f'W2_{str(START)}.pt')
learning_rate = 0.001
curr_idx = 0

print(f'Length of idx_pairs: {len(idx_pairs)}')

lock = threading.Lock()
for epo in range(START, END):
    print(f'Start training epo {epo}...')
    loss_val = 0
    curr_idx = 0

    for data, target in idx_pairs:
        lock.acquire()
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.item()
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()

        curr_idx += 1
        if curr_idx % 100000 == 0:
            print(f'{curr_idx} idx pair finished')
        lock.release()
    print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')

    W1_filename = f'W1_{str(epo+1)}.pt'
    torch.save(W1, W1_filename, _use_new_zipfile_serialization=False)

    W2_filename = f'W2_{str(epo+1)}.pt'
    torch.save(W2, W2_filename, _use_new_zipfile_serialization=False)

    with open('loss.txt', 'a') as lossf:
        lossf.write(str(loss_val/len(idx_pairs)) + '\n')

    if epo != END-1:
        time.sleep(60)

