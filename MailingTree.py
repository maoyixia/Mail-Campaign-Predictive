## This script builds a classification tree from the data preprocessed in DataLoading.py
##     The imported packages need to have been installed on your computer -- see tutorial

# NB: This script is basically an except from the Classification.py
# tutorial, focusing on only the tree-induction functionality and a simple in-sample evaluation

# all imports from Classification.py are included for convenience (this script doesn't use them all)
import numpy as np
import scipy as sp
import sklearn
import os
from sklearn.cross_validation import KFold
from sklearn.cross_validation import ShuffleSplit
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pickle
import csv
import sys

##############
#Loading (deserializing) the data:
# Data were put in the correct format in DataLoading.py
# (This will be much faster than doing all the preprocessing every time we
#  want to do more exploration or modeling.)

def pickleLoad(inputName):
    pk1_file = open(inputName+'.pk1', 'rb')
    pyObj = pickle.load(pk1_file)
    return pyObj

dataMat = pickleLoad('myDataMat')
target = pickleLoad('myTarget')
headers = pickleLoad('myHeaders') # these are the column names in the expanded data
valDict = pickleLoad('myValues') 
num_names = pickleLoad('numericalVariables')
cat_names = pickleLoad('categoricalVariables')

target_data = np.ravel(target) # make it a row vector

#print dataMat

###############
###############
#Now, build a Classification Tree:

#clf is the inducer; thisFit is the particular model built from these training data
clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf = 1, max_depth=20)
thisFit = clf.fit(dataMat, target_data)

# That's it!

##############
##############

#How accurate is the tree, when predicting on the training data?

#get predicted values and class probabilities
myPredictions = thisFit.predict(dataMat) #<-- note: predicting on the training data
correctClass = accuracy_score(target_data, myPredictions)

print "Accuracy on training data: ", correctClass

sys.exit()
################
################

# If you want to visualize the tree:
# Write the tree to a .dot file, in order to visualize it:
#  You'll have to install graphviz on your computer -- just google "install graphviz"

with open("tree.dot", 'w') as f:
     f = tree.export_graphviz(thisFit, out_file=f, feature_names=headers)

print "To view the tree type:  (NB: need to have installed graphviz)"
print "   dot -Tpdf tree.dot -o tree.pdf"
print "Then view the file tree.pdf"

