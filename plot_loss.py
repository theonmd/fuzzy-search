import numpy as np
from matplotlib import pyplot as plt


with open('loss.txt', 'r') as f:
    loss_txt_arr = f.readlines()

loss_vals = [float(line.strip()) for line in loss_txt_arr]

plt.plot(np.arange(1, 51), loss_vals)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss at Each Training Epoch')
plt.savefig('loss.png')
plt.close()

plt.plot(np.arange(31, 51), loss_vals[30:])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss at Each Training Epoch - Last 20 Epoch')
plt.savefig('loss_last20.png')
plt.close()