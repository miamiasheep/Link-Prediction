def size(iter):
    return(sum(1 for _ in iter))


def mapping_graph(G):
	mapping = {}
	idx = 0
	for node in set(G):
		mapping[node] = idx
		idx += 1

	return mapping