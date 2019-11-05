import os
import csv
import argparse
import random
import matplotlib.pyplot as plt
from util import size, mapping_graph
from evaluation import Judge
import model
import networkx as nx
import itertools

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
    parser.add_argument('--goal', type=str, help='auc/f1')
    parser.add_argument('--draw', type=int, help='0~N (N is sample of nodes)', default=0)
    args = parser.parse_args()
    
    inputs = args.input.split(',')
    goal = args.goal
    output_file = open('result/{0}.csv'.format(goal), 'w')

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
        judge = Judge(G)
        
        # grid search for best parameter 
        alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
        pr.grid_search(judge.G_train, judge, goal, alphas)
        
        betas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
        mapping = mapping_graph(judge.G_train)
        G_train_indexed = nx.Graph()
        for edge in judge.train:
            G_train_indexed.add_edge(mapping[edge[0]], mapping[edge[1]])
        katz.grid_search(G_train_indexed, judge, goal, betas, mapping)
        rwr.grid_search(G_train_indexed, judge, goal, betas, mapping)

        # Evaluate Model
        for cur_model in models:
            print('we are evalute {0} model ...'.format(cur_model.name()))
            score = judge.evaluate(cur_model, goal=goal, mapping = mapping)
            output_file.write(',{0}'.format(score))
            print('=' * 50)
        output_file.write('\n')

    
    