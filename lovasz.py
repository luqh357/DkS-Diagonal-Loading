import numpy as np
import scipy as sp
import networkx as nx
from build_graph import *

def phi(v, d, k, alpha, mu):
    return np.sum(np.maximum(np.minimum(v + alpha * (d - mu), 1), 0)) - k

def proximal(v, d, k, alpha):
    mu_u = np.max(d + v / alpha)
    mu_l = np.min(d - (1 - v) / alpha)

    tol = 1e-6

    while(mu_u - mu_l > tol):
        mu_m = (mu_l + mu_u) / 2
        if(phi(v, d, k, alpha, mu_m) * phi(v, d, k, alpha, mu_l) < 0):
            mu_u = mu_m
        else:
            mu_l = mu_m

    return np.maximum(np.minimum(v + alpha * (d - mu_m), 1), 0)

def shrinkage(a, w, kappa):
    return np.maximum(0, a - kappa * w) - np.maximum(0, -a - kappa * w)


def lovasz(file_name, k, t_max):
    G = build_graph(file_name)
    A = nx.adjacency_matrix(G)
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)
    E = nx.incidence_matrix(G, oriented = True)
    L = nx.laplacian_matrix(G)
    d = [val for (_, val) in G.degree()]
    d = np.array(d)
    w = np.ones(m)

    k = int(k)
    t_max = int(t_max)

    rho = 0.1
    Lambda = 1 / rho
    mu = Lambda / (sp.sparse.linalg.svds(E, k = 1, return_singular_vectors = False) ** 2)
    gamma = mu / Lambda
    alpha = 1.8

    x = np.zeros(n)
    
    idx = np.argpartition(d, -k)[-k:]
    x[idx] = 1

    z = E.T @ x
    u = np.zeros(m)

    for t in range(t_max):
        x = proximal(x - gamma * (L @ x - E @ (z - u)), d, k, mu)

        z_old = z
        x_hat = alpha * (E.T @ x) + (1 - alpha) * z_old
        z = shrinkage(x_hat + u, w, Lambda)

        u = u + x_hat - z

        if t == 0:
            x_avg = x
        else:
            x_avg = ((t - 1) * x_avg + x) / t

    x = np.zeros(n)
    idx = np.argpartition(x_avg, -k)[-k:]
    x[idx] = 1

    return x @ A @ x