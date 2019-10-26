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
    parser.add_argument('--goal', type=str, help='auc/f1')
    args = parser.parse_args()
    
    inputs = args.input.split(',')
    goal = args.goal
    output_file = open('result/{0}.csv'.format(goal), 'w')
    
    # pagerank is a dynamic model with parameter
    pr = model.PageRank()
    models = [model.CommonNeighbor, model.Jaccard, model.AdamicAdar, model.PreferentialAttachment, model.TotalNeighbors, pr]
    
    # write header
    for cur_model in models:
        output_file.write(',' + cur_model.name())
    output_file.write('\n')
    
    for input in inputs:
        random.seed(0)
        # write column
        output_file.write(os.path.basename(input).split('.')[0])
        
        # Get the whole graph
        G = construct_graph_with_relation(input, verbose=True)
        data_analysis(G)
        judge = Judge(G)
        
        # grid search for best parameter 
        alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
        pr.grid_search(G, judge, goal, alphas)
        
        # Evaluate Model
        for cur_model in models:
            print('we are evalute {0} model ...'.format(cur_model.name()))
            score = judge.evaluate(cur_model, goal=goal)
            output_file.write(',{0}'.format(score))
            print('=' * 50)
        output_file.write('\n')

    
    