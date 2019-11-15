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
        prediction_list.sort(reverse=True)
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
    parser.add_argument('--goal', type=str, help='auc/f1', default='auc')
    parser.add_argument('--dup', type=str, help='yes/no', default='no')
    args = parser.parse_args()

    base = 'libmf'
    data_base = 'mf'

    inputs = ['Celegans','facebook','NS','PB','Power','Router','USAir','Yeast']
    if args.dup == 'yes':
        output = open(os.path.join('result', 'mf_dup_{0}.csv'.format(args.goal)), 'w')
    else:
        output = open(os.path.join('result', 'mf_{0}.csv'.format(args.goal)), 'w')
    for input in inputs:
        print(input)
        # duplicate training data
        base_name = os.path.join(data_base, input)
        if args.dup == 'yes':
            dup = open(base_name + '.dup', 'w')
            for line in open('{0}.train'.format(base_name)):
                dup.write(line)
                words = line.split()
                dup.write('{} {} {}\n'.format(words[1], words[0], words[2]))
            dup.close()
            run('{0} -f 10 -t 300 -k 15 --quiet {1}.dup {1}.model'.format(os.path.join(base, 'mf-train'), os.path.join(data_base, input)))
        else:
            run('{0} -f 10 -t 300 -k 15 --quiet {1}.train {1}.model'.format(os.path.join(base, 'mf-train'), os.path.join(data_base, input)))
        run('{0} -e 12 {1}.test {1}.model {1}.pred'.format(os.path.join(base, 'mf-predict'), os.path.join(data_base, input)))
        score = evaluate(os.path.join(data_base, input) + '.pred', os.path.join(data_base, input) + '.test', args.goal)
        output.write('{0}, {1}\n'.format(input, score))
        print(evaluate(os.path.join(data_base, input) + '.pred', os.path.join(data_base, input) + '.test', args.goal))
