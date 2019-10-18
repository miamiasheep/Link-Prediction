import math

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

# the sum of the inverse log of the degree of common neighbors. commonly used for social network
class AdamicAdar(Model):
    def name():
        return 'Adamic Adar'

    def predict(G, node1, node2):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        cn_set = n1.intersection(n2)
        return sum(( 1/math.log(sum(1 for _ in G.neighbors(i))) for i in cn_set ))

# the product of the degree of each node
class PreferentialAttachment(Model):
    def name():
        return 'Preferential Attachment'

    def predict(G, node1, node2):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        return len(n1)*len(n2)

# total unique neighbors of the two nodes
class TotalNeighbors(Model):
    def name():
        return 'Total Neighbors'

    def predict(G, node1, node2):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        return len(n1.union(n2))

        










