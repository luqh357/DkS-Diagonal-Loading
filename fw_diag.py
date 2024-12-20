import numpy as np
import scipy as sp
import networkx as nx
from build_graph import *

def fw_diag(file_name, k, l, t_max):
    G = build_graph(file_name)
    n = nx.number_of_nodes(G)
    A = nx.adjacency_matrix(G)
    A = A.astype(float)

    k = int(k)
    l = int(l)
    t_max = int(t_max)

    x = np.ones(n) * (k / n)

    A_diag_load = A + l * sp.sparse.identity(n) # diagonal loading
    L = abs(sp.sparse.linalg.eigsh(A_diag_load, k = 1, which = 'LM', return_eigenvectors = False)) # compute the Lipschitz constant

    for t in range(t_max):
        grad_f = A_diag_load @ x # compute gradient
        s = np.zeros(n)
        idx = np.argpartition(grad_f, -k)[-k:] # choose top k elements
        s[idx] = 1
        d = s - x # compute the FW update direction
        if grad_f @ d == 0: # check if the FW gap is 0
            break
        gamma = min(1, (grad_f @ d) / (L * np.power(np.linalg.norm(d), 2))) # compute the step size
        x = x + gamma * d

    y = np.zeros(n)
    idx = np.argpartition(x, -k)[-k:]
    y[idx] = 1 # post-processing
    
    return y @ A @ y