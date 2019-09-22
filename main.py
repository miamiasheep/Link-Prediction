import argparse
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
    for line in open(args.input):
        words = line.split()
        if count % 1000000 == 0 and verbose:
            print('{} lines processed'.format(count))
        G.add_edge(words[0], words[1])
        count += 1
    return G

def common_neighbor_index(G, node_1, node_2):
    '''
    args:
        1. G: networkx Graph
        2. node_1: index name of node_1
        3. node_2: index_name of node_2
    return:
        dictionary:{cn: common neighbors count, jaccard: jaccard index}
    '''
    n1 = set(G.neighbors(node_1))
    n2 = set(G.neighbors(node_2))
    int_count = len(n1.intersection(n2))
    union_count = len(n1.union(n2))
    ret = {'cn': int_count, 'jaccard': float(int_count) / union_count}
    return ret
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Enter Input File')
    args = parser.parse_args() 
    G = construct_graph_with_relation(args.input, verbose=True)
    
    # calculate common neighbor index of node '1' and node '2'
    ret = common_neighbor_index(G, '1', '2')
    print('common neighbor: {}'.format(ret['cn']))
    print('jaccard index: {}'.format(ret['jaccard']))
    
    
     

    
    