import torch
import networkx as nx
import scipy as sp
from build_graph import *
import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

class Loss(torch.nn.Module):
    def __init__(self, A, k):
        super(Loss, self).__init__()
        self.A = A
        self.k = k

    def forward(self, theta):
        sum_sigmoid = torch.sum(torch.sigmoid(theta))
        exp_phi = torch.nn.functional.relu(sum_sigmoid / self.k - 1)
        x = torch.reciprocal((1 + exp_phi) * (1 + torch.exp(-theta)))
        return -x @ self.A @ x

class Model(torch.nn.Module):
    def __init__(self, init):
        super(Model, self).__init__()
        self.theta = torch.nn.Parameter(init)

def para_adamw(file_name, k, l, t_max, lr):
    G = build_graph(file_name)
    n = nx.number_of_nodes(G)
    A = nx.adjacency_matrix(G)
    A_diag_load = A + l * sp.sparse.identity(n) # diagonal loading
    A = torch.sparse_csr_tensor(torch.tensor(A.indptr), torch.tensor(A.indices), torch.tensor(A.data, dtype=torch.float32))
    A_diag_load = torch.sparse_csr_tensor(torch.tensor(A_diag_load.indptr), torch.tensor(A_diag_load.indices), torch.tensor(A_diag_load.data, dtype=torch.float32))
    
    k = int(k)
    t_max = int(t_max)
    
    model = Model(torch.zeros(n))
    loss = Loss(A_diag_load, k)

    optimizer = torch.optim.AdamW([model.theta], lr) # choose optimizer

    sum_sigmoid = torch.sum(torch.sigmoid(model.theta))
    exp_phi = torch.nn.functional.relu(sum_sigmoid / k - 1)
    x = torch.reciprocal((1 + exp_phi) * (1 + torch.exp(-model.theta))) # parameterize x

    for t in range(t_max):
        optimizer.zero_grad()
        loss_fn = loss(model.theta)
        loss_fn.backward()
        optimizer.step()

    sum_sigmoid = torch.sum(torch.sigmoid(model.theta))
    exp_phi = torch.nn.functional.relu(sum_sigmoid / k - 1)
    x = torch.reciprocal((1 + exp_phi) * (1 + torch.exp(-model.theta)))
    
    y = torch.zeros(n)
    idx = torch.argsort(x)[-k:]
    y[idx] = 1
    obj = y @ A @ y # post-processing

    return obj.detach().numpy()