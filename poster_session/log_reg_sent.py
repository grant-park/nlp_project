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
				self.data.append([self.target_dict[target], curr_tweet_token_count, self.stance_dict[stance]])
				self.total_count += 1

		print 'Data read.'
		print 'There are %s valid data points' % self.total_count
		print 'There are %s tokens' % len(self.token_count)
		for target in self.targets:
			print 'For target %s there are %s data points' % (target, self.target_counts[target])
			for stance in self.stances:
				print '\t For stance %s there are %s data points' % (stance, self.stance_counts[target][stance])
		

	def train(self, target, c=1e5):
		# generate features
		self.labels = []
		self.training_data = []
		for item in self.data:
			if item[0] == self.target_dict[target]:
				self.labels.append(item[2])
				curr_features = []
				for token, count in self.token_count.iteritems():
					curr_features.append(item[1][token])
				self.training_data.append(curr_features)
		self.training_data = np.array(self.training_data)
		self.labels = np.array(self.labels)

		print 'Features generated.'
		print 'Training data has dimension %s times %s' % (self.training_data.shape[0], self.training_data.shape[1])
		print 'Data preparations ready.'

		self.logreg = linear_model.LogisticRegression(C=c)
		self.logreg.fit(self.training_data, self.labels)
		self.model_target = target

		print 'Model trained for target: %s' % target

		

	def predict(self, tweet):
		curr_tweet_token_count = defaultdict(int)
		curr_features = []
		for token in tokenize(tweet):
			curr_tweet_token_count[token] += 1
		for token, count in self.token_count.iteritems():
			curr_features.append(curr_tweet_token_count[token])
		return self.stances[self.logreg.predict(curr_features)]

	def eval(self, target, test_file):
		for_tp = 0
		for_fp = 0 
		for_fn = 0
		for_tn = 0
		against_tp = 0
		against_fp = 0
		against_fn = 0
		against_tn = 0
		with open(test_file, 'r') as f:
			for line in f.read().splitlines()[1:]:
				fields = line.split('\t')
				assert fields[1] in self.targets
				assert fields[3] in self.stances
				if fields[1]==target:
					correct_label = fields[3]
					pred_label = self.predict(fields[2])
					if correct_label=='FAVOR' and pred_label=='FAVOR':
						for_tp += 1
					if correct_label=='FAVOR' and pred_label!='FAVOR':
						for_fn += 1
					if correct_label!='FAVOR' and pred_label=='FAVOR':
						for_fp += 1
					if correct_label!='FAVOR' and pred_label!='FAVOR':
						for_tn += 1
					if correct_label=='AGAINST' and pred_label=='AGAINST':
						against_tp += 1
					if correct_label=='AGAINST' and pred_label!='AGAINST':
						against_fn += 1
					if correct_label!='AGAINST' and pred_label=='AGAINST':
						against_fp += 1
					if correct_label!='AGAINST' and pred_label!='AGAINST':
						against_tn += 1
		if for_tp+for_fp != 0:
			for_precision = for_tp/float(for_tp+for_fp)
		else:
			for_precision = 0
		if for_tp+for_fn != 0:
			for_recall = for_tp/float(for_tp+for_fn)
		else:
			for_recall = 1
		if against_tp+against_fp != 0:
			against_precision = against_tp/float(against_tp+against_fp)
		else:
			against_precision = 0
		if against_tp+against_fn != 0:
			against_recall = against_tp/float(against_tp+against_fn)
		else:
			against_recall = 1
		if for_precision+for_recall!=0:
			f_for = 2*for_precision*for_recall/float(for_precision+for_recall)
		else:
			f_for = 0
		if against_precision+against_recall!=0:
			f_against = 2*against_precision*against_recall/float(against_precision+against_recall)
		else:
			f_against = 0
		return (f_for+f_against)/2.0





if __name__ == '__main__':
	# train on all targets and test
	lr = LogReg()
	lr.prep_data(os.path.join(TRAIN_DIR, TRAIN_FILE))
	accuracies =  []
	for target in lr.targets:
		lr.train(target, 1)
		accuracies.append(lr.eval(lr.model_target, os.path.join(TEST_DIR, TEST_FILE)))
	print accuracies
	plt.xlabel('Target')
	plt.ylabel('Accuracy')
	plt.xticks([0,1,2,3,4], lr.targets)
	plt.plot([0,1,2,3,4], accuracies)
	plt.show()

	# train on one target and test on the rest
	"""lr = LogReg()
	lr.prep_data(os.path.join(TRAIN_DIR, TRAIN_FILE))
	accuracies =  []
	lr.train(lr.targets[3], 1)
	for target in lr.targets:
		accuracies.append(lr.eval(target, os.path.join(TEST_DIR, TEST_FILE)))
	print accuracies
	plt.xlabel('Target')
	plt.ylabel('Accuracy')
	plt.xticks([0,1,2,3,4], lr.targets)
	plt.plot([0,1,2,3,4], accuracies)
	plt.show()"""