#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math

def opendata(file):
	from six.moves import cPickle
	f = open(file, 'rb')
	f.seek(0)
	data = cPickle.load(f, encoding='latin-1')
	f.close()
	return data

data = opendata('data_batch_1')
meta = opendata('batches.meta')

per_fitting = 0.80

datx = data['data']
daty = data['labels']
amount = math.floor(len(data['data'])*per_fitting)
X = datx[:amount]
Y = daty[:amount]
test_points = datx[amount:]
label_points = daty[amount:]

print("Reading data...")

for i in [rnd.randint(0, len(X)) for x in range(0, 3)]:
	c = X[i].reshape((32,32,3), order='F').transpose((1,0,2))
	plt.imshow(c)
	plt.title(meta['label_names'][Y[i]])
	plt.show()

print("Done!\n")

print("Fitting data to classifier. Wait...")
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

print("Done\n")

pred = clf.predict(test_points)

for i in [rnd.randint(0, len(test_points)) for x in range(0, 3)]:
	c = test_points[i].reshape((32,32,3), order='F').transpose((1,0,2))
	plt.imshow(c)
	plt.title(meta['label_names'][pred[i]])
	plt.show()

from sklearn.metrics import accuracy_score
ac = accuracy_score(pred, label_points)
print("The accuracy of the classifier is " + str(ac))