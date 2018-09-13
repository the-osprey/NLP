import sys, argparse
from scipy import sparse
from sklearn import linear_model
from collections import Counter
import numpy as np
import re


# Implement your fancy featurization here
def fancy_featurize(text):
	features = {}

	# adds bag of word representation to features
	features.update(bag_of_words(text))

	# your code here

	return features

# Adds the bag of words representation of the text to feats
def bag_of_words(text):
	word_bag = {}

	words = re.sub("[^\w]", " ", text).split()
	cleaned_words = [w.lower() for w in words]
	for word in cleaned_words:
		d[word] = d.get(word, 0) + 1

	# do stuff here

	return word_bag


# regularization strength to control overfitting (values closer to 0  = stronger regularization)
L2_REGULARIZATION_STRENGTH = {"dumb_featurize": 1, "fancy_featurize": 1 }

# must observe feature at least this many times in training data to include in model
MIN_FEATURE_COUNT = {"dumb_featurize": 10,  "fancy_featurize":10 }





######################################################################
## Don't edit below this line
######################################################################


def dumb_featurize(text):
	feats = {}
	words = text.split(" ")

	for word in words:
		if word == "love" or word == "like" or word == "best":
			feats["contains_positive_word"] = 1
		if word == "hate" or word == "dislike" or word == "worst" or word == "awful":
			feats["contains_negative_word"] = 1

	return feats


class SentimentClassifier:

	def __init__(self, feature_method):
		self.feature_vocab = {}
		self.feature_method = feature_method


	# Read data from file
	def load_data(self, filename):
		data = []
		with open(filename, encoding="utf8") as file:
			for line in file:
				cols = line.split("\t")
				label = cols[0]
				text = cols[1].rstrip()

				data.append((label, text))
		return data

	# Featurize entire dataset
	def featurize(self, data):
		featurized_data = []
		for label, text in data:
			feats = self.feature_method(text)
			featurized_data.append((label, feats))
		return featurized_data

	# Read dataset and returned featurized representation as sparse matrix + label array
	def process(self, dataFile, training = False):
		data = self.load_data(dataFile)
		data = self.featurize(data)

		if training:
			fid = 0
			feature_doc_count = Counter()
			for label, feats in data:
				for feat in feats:
					feature_doc_count[feat]+= 1

			for feat in feature_doc_count:
				if feature_doc_count[feat] >= MIN_FEATURE_COUNT[self.feature_method.__name__]:
					self.feature_vocab[feat] = fid
					fid += 1

		F = len(self.feature_vocab)
		D = len(data)
		X = sparse.dok_matrix((D, F))
		Y = np.zeros(D)
		for idx, (label, feats) in enumerate(data):
			for feat in feats:
				if feat in self.feature_vocab:
					X[idx, self.feature_vocab[feat]] = feats[feat]
			Y[idx] = 1 if label == "pos" else 0

		return X, Y

	def load_test(self, dataFile):
		data = self.load_data(dataFile)
		data = self.featurize(data)

		F = len(self.feature_vocab)
		D = len(data)
		X = sparse.dok_matrix((D, F))
		Y = np.zeros(D, dtype = int)
		for idx, (data_id, feats) in enumerate(data):
			# print (data_id)
			for feat in feats:
				if feat in self.feature_vocab:
					X[idx, self.feature_vocab[feat]] = feats[feat]
			Y[idx] = data_id

		return X, Y

	# Train model and evaluate on held-out data
	def evaluate(self, trainX, trainY, devX, devY):
		(D,F) = trainX.shape
		self.log_reg = linear_model.LogisticRegression(C = L2_REGULARIZATION_STRENGTH[self.feature_method.__name__])
		self.log_reg.fit(trainX, trainY)
		training_accuracy = self.log_reg.score(trainX, trainY)
		development_accuracy = self.log_reg.score(devX, devY)
		print("Method: %s, Features: %s, Train accuracy: %.3f, Dev accuracy: %.3f" % (self.feature_method.__name__, F, training_accuracy, development_accuracy))


	# Predict labels for new data
	def predict(self, testX, idsX):
		predX = self.log_reg.predict(testX)

		out = open("%s_%s" % (self.feature_method.__name__, "predictions.csv"), "w", encoding="utf8")
		out.write("Id,Expected\n")
		for idx, data_id in enumerate(testX):
			out.write("%s,%s\n" % (idsX[idx], int(predX[idx])))
		out.close()

	# Write learned parameters to file
	def printWeights(self):
		out = open("%s_%s" % (self.feature_method.__name__, "weights.txt"), "w", encoding="utf8")
		reverseVocab = [None]*len(self.feature_vocab)
		for feat in self.feature_vocab:
			reverseVocab[self.feature_vocab[feat]] = feat

		out.write("%.5f\t__BIAS__\n" % self.log_reg.intercept_)
		for (weight, feat) in sorted(zip(self.log_reg.coef_[0], reverseVocab)):
			out.write("%.5f\t%s\n" % (weight, feat))
		out.close()


# python3 sentiment_classifier.py --train train.txt --dev dev.txt --test test.txt
if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument("--train", required=True)
	ap.add_argument("--dev", required=True)
	ap.add_argument("--test", required=True)

	args = vars(ap.parse_args())

	trainingFile = args["train"]
	evaluationFile = args["dev"]
	testFile = args["test"]

	for feature_method in [dumb_featurize, fancy_featurize]:
		sentiment_classifier = SentimentClassifier(feature_method)
		trainX, trainY = sentiment_classifier.process(trainingFile, training=True)
		devX, devY = sentiment_classifier.process(evaluationFile, training=False)
		testX, idsX = sentiment_classifier.load_test(testFile)
		sentiment_classifier.evaluate(trainX, trainY, devX, devY)
		sentiment_classifier.printWeights()
		sentiment_classifier.predict(testX, idsX)
