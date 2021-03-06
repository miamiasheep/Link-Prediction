import os
import csv
import argparse
import random
import matplotlib.pyplot as plt
from util import size, generate_indexed_graph
from evaluation import Judge
import model
import networkx as nx
import itertools
import pandas as pd 

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
    
def draw_networks(G, n):
    sample = nx.Graph()
    nodes = random.sample(G.nodes(), n)
    for node in nodes:
        for n in nx.neighbors(G, node[0]):
            sample.add_edge(node[0], n)
            for nn in itertools.islice(nx.neighbors(G, n), 10):
                sample.add_edge(n, nn)
    nx.draw(sample)
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Enter Input File')
    parser.add_argument('--goal', type=str, help='auc/acc')
    parser.add_argument('--draw', type=int, help='0~N (N is sample of nodes)', default=0)
    parser.add_argument('--at', type=int, help='0~N(0 is for label size)', default=0)
    args = parser.parse_args()
    
    inputs = args.input.split(',')
    goal = args.goal
    at = args.at
    output_file = open('result/{0}_{1}.csv'.format(goal, at), 'w')

    # pagerank is a dynamic model with parameter
    pr = model.PageRank()
    katz = model.Katz()
    rwr = model.RWR()
    models = [model.CommonNeighbor, model.Jaccard, model.AdamicAdar, model.PreferentialAttachment, model.TotalNeighbors, pr, katz, rwr]

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
        if args.draw > 0:
            draw_networks(G, args.draw)
            exit()
        judge = Judge(G, input)
        # grid search for best parameter 
        alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
        pr.grid_search(judge.G_train, judge, goal, alphas, at)
        betas = [1e-07, 1e-06, 0.00001, 0.0001, 0.001, 0.01, 0.1]
        G_train_indexed, mapping = generate_indexed_graph(judge.G_train)
        katz.grid_search(G_train_indexed, judge, goal, betas, mapping, at)
        betas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
        rwr.grid_search(G_train_indexed, judge, goal, betas, mapping, at)


        
        # Evaluate Model
        for cur_model in models:
            print('we are evalute {0} model ...'.format(cur_model.name()))
            score = judge.evaluate(cur_model, goal=goal, mapping=mapping, at=at)
            output_file.write(',{0}'.format(score))
            print('=' * 50)
        output_file.write('\n')

        # output heuristics_features for training
        pairs = judge.generate_pairs_for_training()
        df = pd.DataFrame()
        for cur_model in models:
            df[cur_model.name()] = judge.generate_heuristics_features_training(cur_model, mapping, pairs[0], pairs[1])
        df['label'] = [0]*len(pairs[0]) + [1]*len(pairs[1])
        file_name = os.path.basename(input).split('.')[0]
        df.to_csv('ml_heuristics/{}_training.csv'.format(file_name), index=False)

        # output heuristics_features for testing
        df = pd.DataFrame()
        for cur_model in models:
            df[cur_model.name()] = judge.generate_heuristics_features_testing(cur_model, mapping)
        df['label'] = judge.generate_labels_test()
        df.to_csv('ml_heuristics/{}_testing.csv'.format(file_name), index=False)







    
    