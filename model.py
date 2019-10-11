class Model:
    def predict():
        pass
    
class CommonNeighbor(Model):
    def name():
        return 'cn'
    
    def predict(G, node1, node2):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        return len(n1.intersection(n2))
        
class Jaccard(Model):
    def name():
        return 'jaccard'
        
    def predict(G, node1, node2):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        int_count = len(n1.intersection(n2))
        union_count = len(n1.union(n2))
        return int_count / union_count