import networkx as nx

def size(iter):
    return(sum(1 for _ in iter))

def generate_indexed_graph(G):
    mapping = {}
    idx = 0
    G_index = nx.Graph()
    for edge in G.edges():
        for i in range(2):
            if edge[i] not in mapping:
                mapping[edge[i]] = idx
                idx += 1
        G_index.add_edge(mapping[edge[0]], mapping[edge[1]])
    return G_index, mapping
