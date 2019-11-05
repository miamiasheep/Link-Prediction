import math
import networkx as nx
from util import size
import numpy as np 

class Model:    
    def predict():
        pass
    
class CommonNeighbor(Model):
    def name():
        return 'CN'
    
    def predict(G, node1, node2, mapping):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        return len(n1.intersection(n2))
        
class Jaccard(Model):
    def name():
        return 'JAC'
        
    def predict(G, node1, node2, mapping):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        int_count = len(n1.intersection(n2))
        union_count = len(n1.union(n2))
        return int_count / union_count

# the sum of the inverse log of the degree of common neighbors. commonly used for social network
class AdamicAdar(Model):
    def name():
        return 'AA'

    def predict(G, node1, node2, mapping):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        cn_set = n1.intersection(n2)
        epi = 0.01
        return sum(( 1/math.log(size(G.neighbors(i)) + epi) for i in cn_set ))

# the product of the degree of each node
class PreferentialAttachment(Model):
    def name():
        return 'PA'

    def predict(G, node1, node2, mapping):
        n1 = set(G.neighbors(node1))
        n2 = set(G.neighbors(node2))
        return len(n1)*len(n2)

# total unique neighbors of the two nodes
class TotalNeighbors(Model):
    def name():
        return 'TN'

    def predict(G, node1, node2, mapping):
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
        
    def predict(self, G, node1, node2, mapping):
        return math.log(self.pr[node1]) + math.log(self.pr[node2])

# Katz Similarity
class Katz:
    def __init__(self):
        self.s = [] # closed form of katz index

    def name(self):
        return "Katz"

    def train(self, G, beta):
        A = np.array(nx.adjacency_matrix(G).todense())
        dim_A = len(A)
        epi = 0.01 # to avoid singular matrix problem
        self.s = np.linalg.inv(np.eye(dim_A)*(1+epi) - beta*A) - np.eye(dim_A)

    def grid_search(self, G, judge, goal, betas, mapping):
        max_score = 0
        best_param = betas[0]
        max_s = []

        for beta in betas:
            self.train(G, beta)
            score = judge.evaluate(self, goal=goal, option='valid', mapping=mapping)
            if score > max_score:
                max_score = score
                max_s = self.s
                best_param = beta
        print('best param:{0}'.format(best_param))
        self.s = max_s

    def predict(self, G, node1, node2, mapping):
        return self.s[mapping[node1]][mapping[node2]]

# Random Walk with Restart index
# reference: https://cran.r-project.org/web/packages/linkprediction/vignettes/proxfun.html#random-walk-with-restart-rwr
class RWR:
    def __init__(self):
        self.s = []

    def name(self):
        return "Random Walk with Restart"

    def train(self, G, beta):
        dim_G = len(set(G))
        transition_matrix = [[0]*dim_G for _ in range(dim_G)]
        for node, neighbors in G.adjacency():
            neighbors_size = len(neighbors)
            for neighbor in neighbors:
                transition_matrix[node][neighbor] = 1/neighbors_size

        self.s = beta*np.linalg.inv(np.eye(dim_G) - (1-beta)*np.transpose(transition_matrix))

    def grid_search(self, G, judge, goal, betas, mapping):
        max_score = 0
        best_param = betas[0]
        max_s = []

        for beta in betas:
            self.train(G, beta)
            score = judge.evaluate(self, goal=goal, option='valid', mapping=mapping)
            if score > max_score:
                max_score = score
                max_s = self.s
                best_param = beta
        print('best param:{0}'.format(best_param))
        self.s = max_s

    def predict(self, G, node1, node2, mapping):
        return self.s[mapping[node1]][mapping[node2]] + self.s[mapping[node2]][mapping[node1]]










