import os
import argparse
from sklearn import metrics

def run(cmd):
    print(cmd)
    os.system(cmd)
    
def evaluate(pred_file, test_file, goal='auc'):
    preds = [float(line) for line in open(pred_file)]
    labels = [int(line.split()[2]) for line in open(test_file)]
    if goal == 'auc':
        return metrics.roc_auc_score(labels, preds)
    if goal == 'f1':
        prediction_list = []
        for i in range(len(preds)):
            prediction_list.append((preds[i], labels[i]))
        label_size = len([pred for pred in prediction_list if pred[1] == 1])
        correct = 0
        at = label_size
        for pred in prediction_list[:at]:
            if pred[1] == 1 and pred[0] != 0:
                correct += 1
        f1 = correct / at
        return f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', type=str, help='auc/f1')
    args = parser.parse_args()

    base = 'libmf'
    data_base = 'mf'

    inputs = ['Celegans','facebook','NS','PB','Power','Router','USAir','Yeast']
    output = open(os.path.join('result', 'mf_{0}.csv'.format(args.goal)), 'w')
    for input in inputs:
        print(input)
        run('{0} -f 10 -t 100 -l2 0 --quiet {1}.train {1}.model'.format(os.path.join(base, 'mf-train'), os.path.join(data_base, input)))
        run('{0} -e 12 {1}.test {1}.model {1}.pred'.format(os.path.join(base, 'mf-predict'), os.path.join(data_base, input)))
        auc = evaluate(os.path.join(data_base, input) + '.pred', os.path.join(data_base, input) + '.test')
        output.write('{0}, {1}\n'.format(input, auc))
        print(evaluate(os.path.join(data_base, input) + '.pred', os.path.join(data_base, input) + '.test', args.goal))
