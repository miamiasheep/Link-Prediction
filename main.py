import argparse
from evaluation import Judge
import model
import networkx as nx

# The sample size of testing set
SAMPLE_SIZE = 10000
# the k in F1@k, recall@k, precision@k
AT = 100

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Enter Input File')
    args = parser.parse_args()
    
    G = construct_graph_with_relation(args.input, verbose=True)
    
    judge = Judge(G)
    judge.sample(SAMPLE_SIZE)
    models = [model.CommonNeighbor, model.Jaccard, model.AdamicAdar, model.PreferentialAttachment, model.TotalNeighbors]
    for model in models:
        print('we are evalute {0} model ...'.format(model.name()))
        metrics = judge.evaluate(model, AT)
        print('=' * 50)

    
    