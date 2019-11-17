import math
import random
import networkx as nx
import numpy as np
from sklearn import metrics

class Judge:
    def __init__(self, G):
        self.samples = []
        self.G = G
        self.nodes = [n for n in G.nodes()]
        self.edges = [e for e in G.edges()]
        self.edges_set = set(self.edges)
        self.divide_into_train_valid_testing_set()
        self.create_train_graph()
        self.edges_set_train = set(self.train)
    
    def sample_negatives(self, num_samples):
        ans = []
        count = 0
        while count < num_samples:
            cand = random.sample(self.nodes, 2)
            neg_edge = (cand[0], cand[1])
            if neg_edge not in self.edges_set:
                ans.append(neg_edge)
                count += 1
        return ans
    
    def divide_into_train_valid_testing_set(self, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
        random.shuffle(self.edges)
        # divide into train, valid, test
        train_index = math.floor(len(self.edges) * train_ratio)
        valid_index = math.floor(len(self.edges) * valid_ratio) + train_index
        self.train = self.edges[:train_index]
        
        # validation set contains 1:1 negative edge sampling
        self.valid = self.edges[train_index:valid_index]
        neg_samples = self.sample_negatives(len(self.valid))
        for neg_sample in neg_samples:
            self.valid.append(neg_sample)
        self.test = self.edges[valid_index:]
        neg_samples = self.sample_negatives(len(self.test))
        for neg_sample in neg_samples:
            self.test.append(neg_sample)
    
    def create_train_graph(self):
        self.G_train = nx.Graph()
        for edge in self.train:
            self.G_train.add_edge(edge[0], edge[1])



    # 1:1 negative, positive
    def generate_pairs_for_training(self):
        neg = []
        count = 0
        num_samples = len(self.train)
        node_train = set([node for node in self.G_train.nodes()])
        while count < num_samples:
            cand = random.sample(node_train, 2)
            neg_edge = (cand[0], cand[1])
            if neg_edge not in self.edges_set_train:
                neg.append(neg_edge)
                count += 1

        return [neg, self.train]


    def generate_heuristics_features_training(self, model, mapping, negative_pairs, positive_pairs):
        length = len(negative_pairs)
        neg_score = []
        pos_score = []
        for node1, node2 in negative_pairs:
            score = model.predict(self.G_train, node1, node2, mapping)
            neg_score.append(score)
        for node1, node2 in positive_pairs:
            score = model.predict(self.G_train, node1, node2, mapping)
            pos_score.append(score)
        return neg_score+pos_score


    def generate_heuristics_features_testing(self, model, mapping):
        scores = []
        sample = self.test
        node_train = set([node for node in self.G_train.nodes()])
        for node1, node2 in sample:
            if node1 not in node_train or node2 not in node_train:
                continue
            score = model.predict(self.G_train, node1, node2, mapping)
            scores.append(score)
        return scores


    def generate_labels_test(self):
        labels = []
        sample = self.test
        node_train = set([node for node in self.G_train.nodes()])
        for node1, node2 in sample:
            if node1 not in node_train or node2 not in node_train:
                continue
            label = 1 if (node1, node2) in self.edges_set else 0
            labels.append(label)

        return labels



    def evaluate(self, model, option='test', goal='auc', mapping=None):
        prediction_list = []
        if option == 'test':
            sample = self.test
        elif option == 'valid':
            sample = self.valid
        else:
            print('wrong option should be test/valid')
        node_train = set([node for node in self.G_train.nodes()])
        for node1, node2 in sample:
            if node1 not in node_train or node2 not in node_train:
                continue
            score = model.predict(self.G_train, node1, node2, mapping)
            label = 1 if (node1, node2) in self.edges_set else 0
            prediction_list.append([score,label])
        # python's default will use the first element to sort
        prediction_list.sort(reverse = True)
        if goal == 'auc':
            # calculate AUC
            pred = [p[0] for p in prediction_list]
            label = [p[1] for p in prediction_list]
            auc = metrics.roc_auc_score(label, pred)
            print('AUC: {0}'.format(auc))        
            return auc
        elif goal == 'f1':
            # calculate f1 score
            label_size = len([pred for pred in prediction_list if pred[1] == 1])
            correct = 0
            at = label_size
            for pred in prediction_list[:at]:
                if pred[1] == 1 and pred[0] != 0:
                    correct += 1
            f1 = correct / at
            print('F1: {0}'.format(f1))
            return f1
            
