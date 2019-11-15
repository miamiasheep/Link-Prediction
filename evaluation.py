import math
import random
import os
import networkx as nx
import numpy as np
from sklearn import metrics

class Judge:
    def __init__(self, G, input):
        self.samples = []
        self.G = G
        self.input = os.path.basename(input).split('.')[0]
        self.nodes = [n for n in G.nodes()]
        self.edges = [e for e in G.edges()]
        self.edges_set = set(self.edges)
        self.divide_into_train_valid_testing_set()
        self.create_train_graph()
        self.export_data_to_mf_format()
    
    def export_data_to_mf_format(self):
        self.node_train = set([node for node in self.G_train.nodes()])

        if not os.path.isdir('mf'):
            os.mkdir('mf')
        train_file =  open(os.path.join('mf', self.input + '.train'), 'w')
        valid_file = open(os.path.join('mf', self.input + '.valid'), 'w')
        test_file = open(os.path.join('mf', self.input + '.test'), 'w') 
        for edge in self.train:
            if edge[0] not in self.node_train or edge[1] not in self.node_train:
                continue
            train_file.write('{0} {1} 1\n'.format(edge[0], edge[1]))
        for edge in self.valid:
            if edge[0] not in self.node_train or edge[1] not in self.node_train:
                continue
            valid_file.write('{0} {1} 1\n'.format(edge[0], edge[1]))
        for edge in self.valid_neg:
            if edge[0] not in self.node_train or edge[1] not in self.node_train:
                continue
            valid_file.write('{0} {1} 0\n'.format(edge[0], edge[1]))
        for edge in self.test:
            if edge[0] not in self.node_train or edge[1] not in self.node_train:
                continue
            test_file.write('{0} {1} 1\n'.format(edge[0], edge[1]))
        for edge in self.test_neg:
            if edge[0] not in self.node_train or edge[1] not in self.node_train:
                continue
            test_file.write('{0} {1} 0\n'.format(edge[0], edge[1]))
            
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
        self.valid_neg = self.sample_negatives(len(self.valid))
        
        self.test = self.edges[valid_index:]
        self.test_neg = self.sample_negatives(len(self.test))
    
    def create_train_graph(self):
        self.G_train = nx.Graph()
        for edge in self.train:
            self.G_train.add_edge(edge[0], edge[1])

    
    def evaluate(self, model, option='test', goal='auc', mapping=None, at=0):
        prediction_list = []
        if option == 'test':
            sample = self.test
            for neg_sample in self.test_neg:
                sample.append(neg_sample)
        elif option == 'valid':
            sample = self.valid
            for neg_sample in self.valid_neg:
                sample.append(neg_sample)
        else:
            print('wrong option should be test/valid')
        for node1, node2 in sample:
            if node1 not in self.node_train or node2 not in self.node_train:
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
        elif goal == 'acc':
            # calculate acc score
            if at == 0:
                label_size = len([pred for pred in prediction_list if pred[1] == 1])
                at = label_size
            correct = 0
            for pred in prediction_list[:at]:
                if pred[1] == 1 and pred[0] != 0:
                    correct += 1
            acc = correct / at
            print('acc: {0}'.format(acc))
            return acc
            
