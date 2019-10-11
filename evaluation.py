import random

class Judge:
    def __init__(self, G):
        self.samples = []
        self.G = G
        self.nodes = [n for n in G.nodes()]
        self.edges = set([e for e in G.edges()])
        
    def sample(self, sample_size):
        n = len(self.nodes)
        # sample pairs
        self.samples = random.sample([(i, j) for i in range(n) for j in range(i+1, n)], sample_size)
    
    def evaluate(self, model, at):
        prediction_list = []
        for i, j in self.samples:
            score = model.predict(self.G, self.nodes[i], self.nodes[j])
            label = 1 if (self.nodes[i], self.nodes[j]) in self.edges else 0
            prediction_list.append([score,label])
    
        # python's default will use the first element to sort
        prediction_list.sort(reverse = True)
        label_size = len([pred for pred in prediction_list if pred[1] == 1])
        correct = 0
        for pred in prediction_list[:at]:
            if pred[1] == 1:
                correct += 1

        # python3 use floating point devision as default
        precision = correct / at
        recall = correct / label_size
        f1 = 2 * precision * recall / (precision + recall)
        
        print('precision: {0}'.format(precision))
        print('recall: {0}'.format(recall))
        print('f1: {0}'.format(f1))
        return {'precision': precision, 'recall': recall, 'f1': f1}