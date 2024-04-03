import torch

batch_size = 1
# n_1,n_2=31/32,1/32
n_1,n_2=1,0
l1reg, l2reg = 1, 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")