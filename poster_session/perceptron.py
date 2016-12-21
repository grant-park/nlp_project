import numpy as np
import pandas as pd
import sys,random,os,csv, random
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn import linear_model
from sklearn import neural_network

from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

def dict_subtract(vec1, vec2):
    """treat vec1 and vec2 as dict representations of sparse vectors"""
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2: 
        if k not in out:
            out[k] = 0
        out[k] -= vec2[k]
    return dict(out)

def dict_add(vec1, vec2):
    out = defaultdict(float)
    out.update(vec1)
    for k in vec2:
        if k not in out:
            out[k] = 0
        out[k] += vec2[k]
    return dict(out)

def dict_dotprod(d1, d2):
    """Return the dot product (aka inner product) of two vectors, where each is
    represented as a dictionary of {index: weight} pairs, where indexes are any
    keys, potentially strings.  If a key does not exist in a dictionary, its
    value is assumed to be zero."""
    smaller = d1 if len(d1)<len(d2) else d2  # BUGFIXED 20151012
    total = 0
    for key in smaller.keys():
        total += d1.get(key,0) * d2.get(key,0)
    return total

def dict_const_mult(vec, const):
    out = defaultdict(float)
    out.update(vec)
    for k in vec: out[k] = out[k] * const
    return dict(out)

def dict_argmax(dct):
    return max(dct.keys(), key=lambda k: dct[k])


def plot_accuracy_vs_iteration(accuracies, num_iter):
    plt.plot(range(num_iter), accuracies)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Num. iterations vs Accuracy')
    plt.show()
    
def plot_label_stats(label_counts):        
    labels = []
    counts = []
    for k in label_counts.keys():
        labels.append(k)
        counts.append(label_counts[k])
        
    print labels
    print counts
    print ""
    print ""
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.show()

class PerceptronSKLearn(object):
    def __init__(self):
        self.targets = ['Atheism', 'Legalization of Abortion', 'Feminist Movement', 'Climate Change is a Real Concern', 'Hillary Clinton']
        self.target_counts = defaultdict(int)
        self.total_count = 0
        self.feature_count = defaultdict(int)
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.target_dict = {'Atheism':0, 'Legalization of Abortion':1, 'Feminist Movement':2,
                        'Climate Change is a Real Concern':3, 'Hillary Clinton':4}
        
    def vectorize_data(self, train_file, start=None, end=None):
        with open(train_file, 'r') as f:
            lines = f.read().splitlines()
            start = start if start else 1
            end = end+1 if end else len(lines)
            for line in lines[start:end]:
                fields = line.split('\t')
                self.total_count += 1
                self.target_counts[fields[1]] += 1
                self.X_train.append(fields[2])
                self.y_train.append(self.target_dict[fields[1]])
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
            
    def prep_data(self, train_file, start=None, end=None):
        data = []
        conv_dict = {'Atheism':0, 'Legalization of Abortion':1, 'Feminist Movement':2,
                        'Climate Change is a Real Concern':3, 'Hillary Clinton':4}

        # read data
        with open(train_file, 'r') as f:
            lines = f.read().splitlines()
            start = start if start else 1
            end = end+1 if end else len(lines)
            for line in lines[start:end]:
                fields = line.split('\t')
                self.total_count += 1
                self.target_counts[fields[1]] += 1
                tokens = fields[2].lower().split()
                curr_tweet_feature_count = defaultdict(int)
                
                # unigram counts
                for token in tokens:
                    tt_pair_key = "token=%s_target=%s" % (token, fields[1])
                    self.feature_count[token] += 1
                    self.feature_count[tt_pair_key] += 1
                    curr_tweet_feature_count[token] += 1
                    curr_tweet_feature_count[tt_pair_key] += 1
                    
                # bigram counts
                for i in range(len(tokens) - 1):
                    bigram_key = "token1=%s_token2=%s" % (tokens[i], tokens[i+1])
                    self.feature_count[bigram_key] += 1
                    curr_tweet_feature_count[bigram_key] += 1
                    
#                 # trigram counts
#                 for i in range(len(tokens) - 2):
#                     trigram_key = "token1=%s_token2=%s_token3=%s" % (tokens[i], tokens[i+1], tokens[i+2])
#                     self.feature_count[trigram_key] += 1
#                     curr_tweet_feature_count[trigram_key] += 1
                    
                data.append([conv_dict[fields[1]], curr_tweet_feature_count])

        # generate features
        print "Number of training tweets", self.total_count
        print "Number of features", len(self.feature_count)
        self.labels = []
        self.training_data = []
        for item in data:
            self.labels.append(item[0])
            curr_features = []
            for token, count in self.feature_count.iteritems():
                curr_features.append(item[1][token])
            self.training_data.append(curr_features)
        self.training_data = np.array(self.training_data)
        self.labels = np.array(self.labels)
    
    def train_pipeline(self, num_iter=10, eta=0.01):
        self.base_clf = linear_model.Perceptron(n_iter=num_iter, eta0=eta)
        self.clf = Pipeline([
            ('vectorizer', CountVectorizer(analyzer='word', ngram_range=(1, 2), decode_error='ignore')),
            ('tfidf', TfidfTransformer()),
            ('clf', OneVsRestClassifier(self.base_clf))])
        self.clf.fit(self.X_train, self.y_train)
        
    def predict(self, tweet):
        curr_tweet_feature_count = defaultdict(int)
        curr_features = []
        tokens = tweet.split(' ')
        for token in tokens:
            curr_tweet_feature_count[token] += 1
            
        for i in range(len(tokens) - 1):
            bigram_key = "token1=%s_token2=%s" % (tokens[i], tokens[i+1])
            curr_tweet_feature_count[bigram_key] += 1
            
#         for i in range(len(tokens) - 2):
#             trigram_key = "token1=%s_token2=%s_token3=%s" % (tokens[i], tokens[i+1], tokens[i+2])
#             curr_tweet_feature_count[trigram_key] += 1
            
        for feature, count in self.feature_count.iteritems():
            curr_features.append(curr_tweet_feature_count[feature])
        return self.targets[self.clf.predict(np.array(curr_features).reshape(1, -1))]
    
    def do_eval(self, test_file, start=None, end=None):
        with open(test_file, 'r') as f:
            lines = f.read().splitlines()
            start = start if start else 1
            end = end+1 if end else len(lines)
            for line in lines[start:end]:
                fields = line.split('\t')
                self.X_test.append(fields[2])
                self.y_test.append(self.target_dict[fields[1]])
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        predicted = self.clf.predict(self.X_test)
        
        return np.sum(predicted == self.y_test) / float(len(self.y_test))


# PATH_TO_DATA = os.path.join(os.getcwd(), 'twitter_dataset')
PATH_TO_DATA = 'twitter_dataset'
TRAIN_DIR, TRAIN_FILE = os.path.join(PATH_TO_DATA, 'train'), "trainingdata-all-annotations.txt"
TEST_DIR, TEST_FILE = os.path.join(PATH_TO_DATA, 'test'), "testdata-taskA-all-annotations.txt"

train_path = os.path.join(TRAIN_DIR, TRAIN_FILE)
test_path = os.path.join(TEST_DIR, TEST_FILE)

##### OWN IMPLEMENTATION #####
# train_tweets, train_gold_labels, train_label_counts, train_stances, train_file_size = process_file(os.path.join(TRAIN_DIR, TRAIN_FILE), end=600)
# test_tweets, test_gold_labels, test_label_counts, test_stances, test_file_size = process_file(os.path.join(TEST_DIR, TEST_FILE))
# print "Training label counts", train_label_counts
# ppn = Perceptron()
# ppn.fit(train_tweets, train_gold_labels)

##### SKLEARN IMPLEMENTATION #####

def limit_accuracy(sizes):
    accuracies = []
    for num_train in sizes:
        skppn = PerceptronSKLearn()
        skppn.vectorize_data(train_path, end=num_train)
        skppn.train_pipeline(num_iter=50, eta=0.1)
        acc = skppn.do_eval(test_path)
        accuracies.append(acc)
    return accuracies

    
#plt.plot(training_size, accuracies)
#plt.xlabel('Training Size')
#plt.ylabel('Accuracy')
#plt.title('Training Size vs Accuracy')
#plt.show()
