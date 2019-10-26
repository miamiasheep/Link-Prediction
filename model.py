import math
import networkx as nx
from util import size

class Model:    
    def predict():
        pass
    
class CommonNeighbor(Model):
    def name():
        return 'CN'
    
    def predict(G, node1, node2):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        return len(n1.intersection(n2))
        
class Jaccard(Model):
    def name():
        return 'JAC'
        
    def predict(G, node1, node2):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        int_count = len(n1.intersection(n2))
        union_count = len(n1.union(n2))
        return int_count / union_count

# the sum of the inverse log of the degree of common neighbors. commonly used for social network
class AdamicAdar(Model):
    def name():
        return 'AA'

    def predict(G, node1, node2):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        cn_set = n1.intersection(n2)
        epi = 0.01
        return sum(( 1/math.log(size(G.neighbors(i)) + epi) for i in cn_set ))

# the product of the degree of each node
class PreferentialAttachment(Model):
    def name():
        return 'PA'

    def predict(G, node1, node2):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        return len(n1)*len(n2)

# total unique neighbors of the two nodes
class TotalNeighbors(Model):
    def name():
        return 'TN'

    def predict(G, node1, node2):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        return len(n1.union(n2))
        
# Global PageRank
class PageRank:
    def __init__(self):
        self.pr = {}
    
    def name(self):
        return 'PG'
    
    def train(self, G, alpha=0.15):
        self.pr = nx.pagerank(G, alpha)
        
    def grid_search(self, G, judge, goal, alphas):
        max_score = 0
        best_param = alphas[0]
        max_pr = {}
        for alpha in alphas:
            self.train(G, alpha)
            score = judge.evaluate(self, goal=goal, option='valid')
            if score > max_score:
                max_score = score
                max_pr = self.pr
                best_param = alpha
        # set the best pr
        print('best param:{0}'.format(best_param))
        self.pr = max_pr
        
    def predict(self, G, node1, node2):
        return math.log(self.pr[node1]) + math.log(self.pr[node2])
        










