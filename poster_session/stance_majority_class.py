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
		self.target_dict = {'Atheism':0, 'Legalization of Abortion':1, 'Feminist Movement':2,
						'Climate Change is a Real Concern':3, 'Hillary Clinton':4}
		self.total_count = 0
		self.token_count = defaultdict(int)
		self.stance_counts = defaultdict(dict)
		self.stances = ['AGAINST', 'FAVOR', 'NONE']
		self.stance_dict = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}

	def prep_data(self, train_file):
		self.data = []
		

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
				assert target in self.targets
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
					self.data.append([self.target_dict[target], curr_tweet_token_count, self.stance_dict[stance]])
					self.total_count += 1

		print 'Data read.'
		print 'There are %s valid data points' % self.total_count
		print 'There are %s tokens' % len(self.token_count)
		for target in self.targets:
			print 'For target %s there are %s data points' % (target, self.target_counts[target])
			for stance in self.stances:
				print '\t For stance %s there are %s data points' % (stance, self.stance_counts[target][stance])
		

	def predict(self, target, tweet):
		max_stance_count = 0
		max_stance = ''
		for stance in self.stances:
			if self.stance_counts[target][stance] > max_stance_count:
				max_stance_count = self.stance_counts[target][stance]
				max_stance = stance
		return max_stance

	def eval(self, target, test_file):
		correct = 0
		incorrect = 0
		with open(test_file, 'r') as f:
			for line in f.read().splitlines()[1:]:
				fields = line.split('\t')
				assert fields[1] in self.targets
				assert fields[3] in self.stances
				if fields[1]==target:
					pred_correct = self.predict(target, fields[2]) == fields[3]
					if pred_correct:
						correct += 1
					else:
						incorrect += 1
		return correct/float(incorrect+correct)





if __name__ == '__main__':
	# train on all targets and test
	lr = LogReg()
	lr.prep_data(os.path.join(TRAIN_DIR, TRAIN_FILE))
	accuracies =  []
	for target in lr.targets:
		accuracies.append(lr.eval(target, os.path.join(TEST_DIR, TEST_FILE)))
	print accuracies
	plt.xlabel('Target')
	plt.ylabel('Accuracy')
	plt.xticks([0,1,2,3,4], lr.targets)
	plt.plot([0,1,2,3,4], accuracies)
	plt.show()