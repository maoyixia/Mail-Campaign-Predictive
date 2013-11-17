## This script loads the tutorial data into Python and saves it in a
## format friendly to the data analytis and visualization packages.
## An alternative would be for each analysis script to begin with code
## such as this, and each time read the data from the .csv file.

## Imported packages need to have been installed on your computer -- see tutorial

import numpy as np
import scipy as sp
from sklearn.feature_extraction import DictVectorizer
import pickle
import sys


##SECTION 1: PREPROCESSING -- load the data and extract meta-data

print 'Loading Data'

## Load data: be careful: first ("header") row is the feature names!  

f = open('mailing_hw3.csv', 'rU')

#create a list containing every line in the file
file_list = f.readlines()

#this is the list of feature names
headers = file_list[0].rstrip().split(',')  #comma separated

#get number of features and number of instances
numFeatures = len(headers)  # This includes the target variable in the feature count
print "In the raw data there are %d features (including the target variable)" % numFeatures
numInstances = len(file_list)-1
print "There are %d instances" % numInstances

print headers
for i in range(1,11):
    print file_list[i].rstrip().split(',')


#First step: decide what variable type each variable is.  This can be
#done by using the data dictionary and/or investigating the .csv file by
#hand.

#Also, decide on which variables to keep/get rid of.  In this example,
#let's get rid of firstdate and lastdate.  This process can be
#followed for any subset of variables that you'd like to keep or get
#rid of.

#numerical variables: avggift, lastgife, pgift, ampergift 
num_names = ['avggift', 'lastgife', 'pgift', 'ampergift']

# #discarded variables: firstdate,lastdate
# #discard_names = ['Firstdate', 'Lastdate','Income', 'pepstrfl', 'rfaa2', 'rfaf2']
# discard_names = ['Firstdate', 'Lastdate']

# #categorical variables: income, rfaf2, rfaa2, pepstrfl
# cat_names = ['Income', 'pepstrfl', 'rfaa2', 'rfaf2']
# #cat_names = []

#target variable: response
target_names = ['response']

print 'Initializing'

#initialize the proper data structures for each type of variable
numericals = np.zeros((numInstances-1, len(num_names)))  #remember, np is numpy
# discards = np.zeros((numInstances-1, len(discard_names)))
# categoricals = []
target = np.zeros((numInstances-1, len(target_names)))

print 'Processing Data'

#Plan of attack: step through file_list one instance at a time
#(skipping the header row).  Place numerical variables, "discard"
#variables, and the target variable in numpy arrays.  Place
#categorical variables in a list of dicts to prepare them for some
#more processing (form special categorical representation).

for i in range(1,numInstances):  #process each instance
    line = file_list[i].rstrip().split(',')  #feature are comma separated
    # D = {}
    numCount = 0
    disCount = 0
    for j in xrange(numFeatures):
        #numerical features
        if num_names.count(headers[j]) > 0:  #look up whether this feature is numeric
            numericals[i-1][numCount] = line[j]
            numCount = numCount+1
        #target variable
        elif target_names.count(headers[j])>0:
            target[i-1][0] = line[j]
        #discard variables
        # elif discard_names.count(headers[j])>0:
        #     #discards[i-1][disCount] = line[j]
        #     disCount = disCount+1
        # #else it's a categorical variable, so put it in the dict for categoricals
        # else:
        #     D[headers[j]] = line[j]
    # #append each dict to the list of dicts
    # categoricals.append(D)

#place numerical data into "dataMat" which may have categorical variables added to it
dataMat = numericals

#record header names
all_headers = num_names

#initialize a dict that will later store the number of values a categorical variable can take on
valDict = {}
for num_name in num_names:
    valDict[num_name] = 1 # numerical variables will be recorded as having one value
# # note that this might cause confusion later
# # if a categorical variable were to have
# # only one value

print valDict;
print dataMat;


# #print categoricals[3] # before transforming
# #Print len(categoricals)
# #print "stopping.."
# #sys.exit() # just stops here

# ## dataset is now in a list of dicts: can use DictVectorize to format categorical variables

# if len(cat_names)>0:
#     print 'Transforming Categorical Variables'

#     vec = DictVectorizer()
#     transformed = vec.fit_transform(categoricals).toarray()

#     #print transformed[3]
#     #sys.exit()

#     #list of new categorical feature names
#     featNames = vec.get_feature_names()

#     print 'Forming final feature set'
#     #concatenate numerical and categorical features
#     dataMat = np.hstack((dataMat, transformed)) # the data matrix

#     #concatenate numerical and categorical header names
#     all_headers = all_headers+featNames

#     #print "New feature set: ", all_headers
#     #sys.exit()

#     print 'Getting categorical feature values'
#     #get the number of values for each categorical feature
    
#     for i in xrange(len(featNames)):
#         #get categorical variable names
#         name = featNames[i].rstrip().split('=')[0]
#         #this counts unique values for each variable name
#         valDict[name] = valDict.setdefault(name,0)+1

#load use data
f_use = open('mailing_hw3_use.csv', 'rU')
file_list_use = f_use.readlines()
numInstances_use = len(file_list_use)-1
numericals_use = np.zeros((numInstances_use-1, len(num_names)))
target_use = np.zeros((numInstances_use-1, len(target_names)))

for i in range(1,numInstances_use):  #process each instance
    line_use = file_list_use[i].rstrip().split(',')  #feature are comma separated
    numCount_use = 0
    for j in xrange(numFeatures):
        #numerical features
        if num_names.count(headers[j]) > 0:  #look up whether this feature is numeric
            numericals_use[i-1][numCount_use] = line_use[j]
            numCount_use = numCount_use+1
        #target variable
        elif target_names.count(headers[j])>0:
            target_use[i-1][0] = line_use[j]

#place numerical data into "dataMat" which may have categorical variables added to it
dataMat_use = numericals_use
print 'This is dataMat_use'
print dataMat_use

print 'This is dataMat'
print dataMat


#we can "serialize" files using pickle.  It's a lot faster than reloading from the csv and preprocessing every time.

def pickleIt(pyName, outputName):
    output = open(outputName+'.pk1', 'wb')
    pickle.dump(pyName, output)
    output.close()

print 'Serializing'

pickleIt(dataMat, 'myDataMat')
pickleIt(target, 'myTarget') 
pickleIt(all_headers, 'myHeaders')
pickleIt(valDict, 'myValues')
pickleIt(num_names, 'numericalVariables')
# pickleIt(cat_names, 'categoricalVariables')
pickleIt(dataMat_use, 'myDataMat_use')
pickleIt(target_use, 'myTarget_use')


print ' '
print 'Printing variable names and indices for future reference'
#print out variable names and indices for future reference
for i in xrange(len(all_headers)):
    print('Index number is '+repr(i)+' and feature name is '+repr(all_headers[i]))



