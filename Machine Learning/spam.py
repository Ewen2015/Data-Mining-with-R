import os
print(os.getcwd())
os.chdir("/Users/ewenwang/Downloads")

import numpy as np

# read spam dataset from file
spam_values = np.genfromtxt('spambase.data', delimiter=',')

# let's take a look at some values
print(spam_values[:6,:4])

# now let's read the column names
fl = open('spambase.names', 'r')
lines = [line.strip() for line in fl]
fl.close()

colnames = [line.partition(':')[0] for line in lines if not (len(line) == 0 or line[0] == '|' or line[0] == '1')]
colnames.append('spam')

import pandas as pd
spam_df = pd.DataFrame(spam_values,columns=colnames)

# finally make the spam labels +1 and -1 since we are going to use this in the knn classifier
spam_df['spam']=2*spam_df['spam']-1

print(spam_df.ix[:3,-3:])

### Data exploration ==========================================================

import math

nsamples = spam_df.shape[0]
ntest = math.floor(.2 * nsamples)
ntune = math.floor(.1 * nsamples)

# we want to make this reporducible so we seed the random number generator
np.random.seed(1)
all_indices = np.arange(nsamples)+1
np.random.shuffle(all_indices)
test_indices = all_indices[:ntest]
tune_indices = all_indices[ntest:(ntest+ntune)]
train_indices = all_indices[(ntest+ntune):]

spam_train = spam_df.ix[train_indices,:]
spam_tune = spam_df.ix[tune_indices,:]
spam_test = spam_df.ix[test_indices,:]

### Data exploration ==========================================================

print(len(spam_train.columns))
print(spam_train.shape)

# what is the mean frequency per label for a few features
grouped_dat = spam_train.groupby('spam').mean()

import pylab as plt

grouped_dat.ix[:,-15:-3].T.plot(kind='bar')
plt.show()

### AdaBoost  ==================================================================

from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
	max_depth=1, random_state=0).fit(X_train, y_train)

clf.score(X_test, y_test) 





























