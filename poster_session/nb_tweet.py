from __future__ import division
from collections import defaultdict
import math
import os

PATH_TO_DATA = '../progress_proposal/twitter_dataset'
TRAIN_DIR, TRAIN_FILE = os.path.join(PATH_TO_DATA, 'train'), "shuffled.txt"
TEST_DIR, TEST_FILE = os.path.join(PATH_TO_DATA, 'test'), "testdata-taskA-all-annotations.txt"

class NB_Baseline:

    def __init__(self):
        self.targets = ['Atheism', 'Legalization of Abortion', 'Feminist Movement', 'Climate Change is a Real Concern', 'Hillary Clinton']
        self.vocab = set()
        self.doc_count_dict = { "Atheism": 0.0, "Legalization of Abortion": 0.0, "Feminist Movement": 0.0, "Climate Change is a Real Concern": 0.0, "Hillary Clinton": 0.0 }
        self.token_count_dict = { "Atheism": 0.0, "Legalization of Abortion": 0.0, "Feminist Movement": 0.0, "Climate Change is a Real Concern": 0.0, "Hillary Clinton": 0.0 }
        self.doc_token_count_dict = { "Atheism": defaultdict(float), "Legalization of Abortion": defaultdict(float), "Feminist Movement": defaultdict(float), "Climate Change is a Real Concern": defaultdict(float), "Hillary Clinton": defaultdict(float) }
        self.total_doc_count = 0

    def train(self, dir, filename, limit=None):
        with open(os.path.join(dir, filename),'r') as doc:
            iterdoc = iter(doc)
            attr = next(iterdoc).split() # differentiate first line
            for index,line in enumerate(iterdoc):
                if limit and index == limit:
                    return
                entry = line.split("\t")
                target = entry[1]
                tweet_content = map(lambda x: x.lower(), entry[2].split())
                self.total_doc_count += 1
                self.doc_count_dict[target] += 1
                self.token_count_dict[target] += len(tweet_content)
                for each in tweet_content:
                    if each not in self.doc_token_count_dict[target]:
                        self.doc_token_count_dict[target][each] = 0
                    self.doc_token_count_dict[target][each] += 1
                    self.vocab.add(each)

    def p_word_given_label_and_psuedocount(self, word, label, alpha):
        return (self.doc_token_count_dict[label][word] + alpha)/(self.token_count_dict[label] + (len(self.vocab)*alpha))

    def log_posterior(self, bag, label, alpha):
        return math.log(self.doc_count_dict[label]/self.total_doc_count) + sum(map(lambda x: math.log(self.p_word_given_label_and_psuedocount(x,label,alpha)), bag))

    def classify(self, bag, alpha):
        for each in list(self.doc_count_dict):
            if self.doc_count_dict[each] == 0:
                del self.doc_count_dict[each]
                self.targets.remove(each)
        return max(map(lambda x: (x, self.log_posterior(bag,x,alpha)), self.targets), key = lambda x: x[1])[0]

    def eval(self, alpha):
        accuracy = 0
        total = 0
        with open(os.path.join(TEST_DIR, TEST_FILE),'r') as doc:
            iterdoc = iter(doc)
            attr = next(iterdoc).split() # differentiate first line
            for index,line in enumerate(iterdoc):
                entry = line.split("\t")
                if entry[1] == self.classify(entry[2].lower(), alpha):
                    accuracy += 1
                total += 1
        return accuracy/total

def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    import matplotlib.pyplot as plt
    plt.plot(psuedocounts, accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')
    plt.show()

if __name__ == '__main__':
    limit = range(50,2501,50)
    accuracies = []
    for i in limit:
        nb = NB_Baseline()
        nb.train(TRAIN_DIR,TRAIN_FILE,i)
        accuracies.append(nb.eval(36))
    import matplotlib.pyplot as plt
    plt.plot(limit, accuracies)
    plt.xlabel('Limit Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Limit Parameter vs. Accuracy Experiment')
    plt.show()
    # Plot
    # psuedocounts = range(1,50)
    # accuracies = map(lambda x: nb.eval(x),psuedocounts)
    # plot_psuedocount_vs_accuracy(psuedocounts, accuracies)
 
