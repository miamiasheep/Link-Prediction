import os
import csv
import argparse
import random
from util import size
from evaluation import Judge
import model
import networkx as nx

def construct_graph_with_relation(input_file_name, verbose=False):
    '''
    args:
        1. input_file_name: the relation txt file
        2. verbose: If you want check progress, set verbose = True
    Return:
        nexworkx Graph
    '''
    G = nx.Graph()
    count = 0
    for line in open(input_file_name):
        words = line.split()
        if count % 1000000 == 0 and verbose:
            print('{} lines processed'.format(count))
        G.add_edge(words[0], words[1])
        count += 1
    return G

def data_analysis(G):
    nodes = size(G.nodes())
    edges = size(G.edges())
    print('nodes: {}'.format(nodes))
    print('edges: {}'.format(edges))
    print('average degree: {}'.format(edges/nodes))
    print('max degree: {}'.format(max(size(G.neighbors(n)) for n in G.nodes())))
    print('min degree: {}'.format(min(size(G.neighbors(n)) for n in G.nodes())))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Enter Input File')
    args = parser.parse_args()
    
    random.seed(0)
    inputs = args.input.split(',')
    auc_file = open('result/auc.csv', 'w')
    f1_file = open('result/f1.csv', 'w')
    models = [model.CommonNeighbor, model.Jaccard, model.AdamicAdar, model.PreferentialAttachment, model.TotalNeighbors]
    # write header
    for model in models:
        auc_file.write(',' + model.name())
        f1_file.write(',' + model.name())
    auc_file.write('\n')
    f1_file.write('\n')
    
    for input in inputs:
        # write column
        auc_file.write(os.path.basename(input).split('.')[0])
        f1_file.write(os.path.basename(input).split('.')[0])
        # Get the whole graph
        G = construct_graph_with_relation(input, verbose=True)
        data_analysis(G)
        # Evaluate Model
        judge = Judge(G)
        for cur_model in models:
            print('we are evalute {0} model ...'.format(cur_model.name()))
            metrics = judge.evaluate(cur_model)
            auc_file.write(',{0}'.format(metrics['AUC']))
            f1_file.write(',{0}'.format(metrics['F1']))
            print('=' * 50)
        auc_file.write('\n')
        f1_file.write('\n')

    
    