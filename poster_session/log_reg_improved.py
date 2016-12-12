import os
from collections import defaultdict
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import re

import warnings
warnings.filterwarnings("ignore")

PATH_TO_DATA = 'twitter_dataset'
TRAIN_DIR, TRAIN_FILE = os.path.join(PATH_TO_DATA, 'train'), "trainingdata-all-annotations.txt"
TEST_DIR, TEST_FILE = os.path.join(PATH_TO_DATA, 'test'), "testdata-taskA-all-annotations.txt"

def tokenize(text):
		text = re.sub('[,#@/&]', ' ', text)
		text = re.sub('[ ]+', ' ', text)
		text = re.sub('^[ ]+', '', text)
		text = re.sub('[ ]+$', '', text)
		text = text.lower()
		return text.split(' ')

class LogReg:
	def __init__(self):
		self.targets = ['Atheism', 'Legalization of Abortion', 'Feminist Movement', 'Climate Change is a Real Concern', 'Hillary Clinton']
		self.target_counts = defaultdict(int)
		self.total_count = 0
		self.token_count = defaultdict(int)
		self.stance_counts = defaultdict(dict)
		self.stances = ['AGAINST', 'FAVOR', 'NONE']

	def prep_data(self, train_file):
		data = []
		self.conv_dict = {'Atheism':0, 'Legalization of Abortion':1, 'Feminist Movement':2,
						'Climate Change is a Real Concern':3, 'Hillary Clinton':4}

		# read data
		with open(train_file, 'r') as f:
			for line in f.read().splitlines()[1:]:
				fields = line.split('\t')
				# get tokens
				tweet = fields[2]
				tokens = tokenize(tweet)
				curr_tweet_token_count = defaultdict(int)
				for token in tokens:
					self.token_count[token] += 1
					curr_tweet_token_count[token] += 1
				# get and validate target
				target = fields[1]
				assert target in self.conv_dict.keys()
				self.target_counts[target] += 1
				# get and validate stance and increment stance counts
				stance = fields[3]
				assert stance in self.stances
				if stance in self.stance_counts[target]:
					self.stance_counts[target][stance] += 1
				else:
					self.stance_counts[target][stance] = 1
				# aggregate data
				if stance!='NONE':
					data.append([self.conv_dict[fields[1]], curr_tweet_token_count, stance])
					self.total_count += 1

		print 'Data read.'
		print 'There are %s valid data points' % self.total_count
		print 'There are %s tokens' % len(self.token_count)
		for target in self.targets:
			print 'For target %s there are %s data points' % (target, self.target_counts[target])
			for stance in self.stances:
				print '\t For stance %s there are %s data points' % (stance, self.stance_counts[target][stance])

		# generate features
		self.labels = []
		self.training_data = []
		for item in data:
			self.labels.append(item[0])
			curr_features = []
			for token, count in self.token_count.iteritems():
				curr_features.append(item[1][token])
			self.training_data.append(curr_features)
		self.training_data = np.array(self.training_data)
		self.labels = np.array(self.labels)

		print 'Features generated.'
		print 'Training data has dimension %s times %s' % (self.training_data.shape[0], self.training_data.shape[1])
		print 'Data preparations ready.'

	def train(self, c=1e5):
		self.logreg = linear_model.LogisticRegression(C=c)
		self.logreg.fit(self.training_data, self.labels)

	def predict(self, tweet):
		curr_tweet_token_count = defaultdict(int)
		curr_features = []
		for token in tokenize(tweet):
			curr_tweet_token_count[token] += 1
		for token, count in self.token_count.iteritems():
			curr_features.append(curr_tweet_token_count[token])
		return self.targets[self.logreg.predict(curr_features)]

	def eval(self, test_file):
		correct = 0
		incorrect = 0
		with open(test_file, 'r') as f:
			for line in f.read().splitlines()[1:]:
				fields = line.split('\t')
				assert fields[1] in self.conv_dict.keys()
				assert fields[3] in self.stances
				if fields[3]!='NONE':
					pred_correct = self.predict(fields[2]) == fields[1]
					if pred_correct:
						correct += 1
					else:
						incorrect += 1
		return correct/float(incorrect+correct)





if __name__ == '__main__':
	lr = LogReg()
	lr.prep_data(os.path.join(TRAIN_DIR, TRAIN_FILE))
	accuracies =  []
	for c in [0.01, 0.1, 0.5, 1, 10, 1e2, 1e3]:
		lr.train(c)
		accuracies.append(lr.eval(os.path.join(TEST_DIR, TEST_FILE)))
	print accuracies
	plt.xlabel('Inverse of regularization strength')
	plt.ylabel('Accuracy')
	plt.xticks([0,1,2,3,4,5,6], ['0.01', '0.1', '0.5', '1', '10', '1e2', '1e3'])
	plt.plot([0,1,2,3,4,5,6], accuracies)
	plt.show()