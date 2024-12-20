import networkx as nx

def build_graph(file_name):
    file = open("./dataset/" + file_name + ".txt", "r")

    G = nx.Graph()
    for line in file:
        if not line.strip().startswith("#"):
            u, v = map(int, line.split())
            G.add_edge(u, v)

    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)

    file.close()

    return G
