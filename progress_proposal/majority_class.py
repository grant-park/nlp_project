import os
from collections import defaultdict

PATH_TO_DATA = 'twitter_dataset'
TRAIN_DIR, TRAIN_FILE = os.path.join(PATH_TO_DATA, 'train'), "trainingdata-all-annotations.txt"
TEST_DIR, TEST_FILE = os.path.join(PATH_TO_DATA, 'test'), "testdata-taskA-all-annotations.txt"

class MajorityClass:
	def __init__(self):
		self.targets = ['Atheism', 'Legalization of Abortion', 'Feminist Movement', 'Climate Change is a Real Concern', 'Hillary Clinton']
		self.target_counts = defaultdict(int)
		self.total_counts = 0
		self.majority_class_pred = ''

	def train(self, train_file):
		with open(train_file, 'r') as f:
			for line in f.read().splitlines()[1:]:
				fields = line.split('\t')
				self.total_counts += 1
				self.target_counts[fields[1]] += 1
		curr_max_target = 0
		for key, value in self.target_counts.iteritems():
			if value > curr_max_target:
				curr_max_target = value
				self.majority_class_pred = key

	def predict(self, tweet):
		return self.majority_class_pred

	def eval(self, test_file):
		correct = 0
		incorrect = 0
		with open(test_file, 'r') as f:
			for line in f.read().splitlines()[1:]:
				fields = line.split('\t')
				pred_correct = self.predict(fields[2]) == fields[1]
				if pred_correct:
					correct += 1
				else:
					incorrect += 1
		return correct/float(incorrect+correct)

	def print_model(self):
		print self.total_counts
		print self.target_counts

if __name__ == '__main__':
	mc = MajorityClass()
	mc.train(os.path.join(TRAIN_DIR, TRAIN_FILE))
	mc.print_model()

	print mc.predict('I love Hillary Clinton')

	print mc.eval(os.path.join(TEST_DIR, TEST_FILE))