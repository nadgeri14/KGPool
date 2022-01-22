import argparse
import math
from sklearn.metrics import f1_score
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='input file')
args = parser.parse_args()

n_class = 705

logloss = 0
n_right = 0
n_instances = 0

y_true = []
y_pred = []

with open(args.input, "r") as f:
    lines = [f.readline().split() for i in range(705)]
    while(lines[0] != []):
        max_score = 0
        arg_max = -1
        right = -1
        right_score = 0
        for line in lines:
            if len(line) != 5:
                #print(line)
                continue
            if (line[2] == "1"): # right answer
                right = int(line[0])
                right_score = float(line[1])

            score = float(line[1])
            if(score > max_score):
                max_score = score
                arg_max = int(line[0])
        
        y_true.append(right)
        y_pred.append(arg_max)
        
        
        if(arg_max == right):
            n_right += 1
        logloss += math.log(right_score + 1e-40)

        n_instances += 1
        lines = [f.readline().split() for i in range(705)]
        

print(n_right/n_instances, logloss/n_instances)
labels = list(set(y_true) - set([1]))
print('f1:', f1_score(y_true, y_pred, labels=labels, average=None))
pickle.dump((y_pred, y_true), open(args.input + ".pkl", "wb"))