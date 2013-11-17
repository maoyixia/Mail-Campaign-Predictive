
## This script gives some example data explorations in Python, using
## the data we preprocessed with DataLoading.py

## Imported packages need to have been installed on your computer -- see tutorial

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import sys

#Deserializing the data: (This will be much faster than doing all the
#preprocessing every time we want to do more exploration or modeling.)

def pickleLoad(inputName):
    pk1_file = open(inputName+'.pk1', 'rb')
    pyObj = pickle.load(pk1_file)
    return pyObj

dataMat = pickleLoad('myDataMat')
target = pickleLoad('myTarget')
headers = pickleLoad('myHeaders')
valDict = pickleLoad('myValues') 
num_names = pickleLoad('numericalVariables')
cat_names = pickleLoad('categoricalVariables')

#Pull out numerical vs. categorical indices - will be very useful
#later!  (Often different sorts of analyses will be done on the
#numerics vs. the categoricals)

#Note: this looks kind of awkward but makes the script more flexible
#for repurposing for other uses (rather than just a demo)

numerical_inds = []
categorical_inds = []
for i in xrange(len(headers)):
    if headers[i] in valDict.keys() and valDict[headers[i]] == 1:
        #numerical variables were recorded as having 1 value in valDict in DataLoading.py
        numerical_inds.append(i) 
    else:
        categorical_inds.append(i)

#subset numerical data for summary statistics
numericals = dataMat[:,numerical_inds]
num_heads = [headers[x] for x in numerical_inds]

#these are numpy functions -- they compute the summary stats for all
#the features in one call to the function
mins = np.min(numericals, axis = 0)
maxes = np.max(numericals, axis = 0)
means = np.mean(numericals, axis=0)
stds = np.std(numericals, axis=0)

for i in xrange(len(num_heads)): #print the stats for each feature
    print('Feature is: '+repr(num_heads[i]))
    print('Min value is: '+repr(mins[i]))
    print('Max value is: '+repr(maxes[i]))
    print('Mean is: '+repr(means[i]))
    print('Std Dev is: '+repr(stds[i]))
    print(' ')

#sys.exit()

if len(cat_names)>0:
    #subset categorical data for summary statistics
    categoricals = dataMat[:,categorical_inds]
    cat_heads = [headers[x] for x in categorical_inds]

    #use numpy sum
    cat_sums = np.sum(categoricals, axis=0)

    for i in xrange(len(cat_heads)):
        print('Cat is: '+repr(cat_heads[i])+' and freq is '+repr(cat_sums[i]))

    #sys.exit()

## PLOTTING/SIMPLE VISUALIZATION

#look at target variable
fig,ax = plt.subplots()
myDisp = ax.hist(target,bins=2)
ax.set_xlabel('class variable')
ax.set_ylabel('Frequency')
ax.set_title('Class value frequencies')
plt.savefig('saved_fig.pdf')  #let's save this figure to a file, before going on
plt.show()

#sys.exit()

##can look at histogram plots of different numerical attributes (quick-n-dirty method)
amount = dataMat[:,0]
fig,ax = plt.subplots()
myDisp = ax.hist(amount,bins = 20)
ax.set_xlabel('Average Donation Amount')
ax.set_ylabel('Number of individuals')
ax.set_title('How much do people donate?')
plt.show()

from matplotlib.ticker import MultipleLocator

##same with average gift amount
gavr = dataMat[:,2]
fig,ax = plt.subplots()
myDisp = ax.hist(gavr,bins = 100)
majorLocator   = MultipleLocator(100)
ax.xaxis.set_major_locator(majorLocator)
ax.set_yscale('symlog')
ax.set_xlabel('Average Donation Amount - when actually donated')
ax.set_ylabel('Number of individuals')
ax.set_title('How much do people donate?')
plt.show()


##example of histogram plot for categorical attribute Income

#need to do a little fiddling with indices to use cat_sums
plottingVar = 'Income' 

#makes sure that plottingVar is actually in the list!
if cat_names.count(plottingVar)>0:
    plottingInd = cat_names.index(plottingVar)

    #figure out where that variable starts and ends
    startingInd = 0
    if plottingInd>0:
        for i in xrange(plottingInd-1):
            startingInd+=valDict[cat_names[i]]
    endingInd = startingInd+valDict[plottingVar]

    #make a bar chart
    ind = np.arange(valDict[plottingVar])
    bars = cat_sums[startingInd:endingInd]
    fig, ax = plt.subplots()
    width=1
    myDisp = ax.bar(ind, bars, width)
    ax.set_xlabel(plottingVar)
    ax.set_ylabel('Frequency')
    plt.show()

    ##Getting a little more fancy..
    ##now look at class by income: stacked bar graph
    ##assuming we use plottingVar as above

    #first need to aggregate, start by initializing np arrays of proper size

    array0 = np.zeros(valDict[plottingVar])
    array1 = np.zeros(valDict[plottingVar])

    #for every instance, add to the correct slot in array0 if target is 0, array1 else
    for i in xrange(categoricals.shape[0]):
        for j in xrange(valDict[plottingVar]):
            arrInd = startingInd+j
            if categoricals[i][arrInd] == 1:
                if target[i] == 1:
                    array1[j]+=1
                elif target[i] == 0:
                    array0[j]+=1

    ##stacked bar graph
    ind = np.arange(valDict[plottingVar])
    disp0 = plt.bar(ind, array0,color='b')
    disp1 = plt.bar(ind, array1,bottom = array0,color='r')
    plt.ylabel('Class value frequencies')
    plt.title('Class value frequency by '+plottingVar)
    plt.legend( (disp0[0], disp1[0]), ('Class = 0', 'Class = 1'))
    plt.show()


    ##bar graph
    fig,ax = plt.subplots()
    width = .35
    rects1 = ax.bar(ind, array0, width, color='b')
    rects2 = ax.bar(ind+width, array1, width, color = 'r')
    ax.set_ylabel('Class value frequency')
    ax.set_title('Class value frequency by '+plottingVar)
    ax.legend( (rects1[0], rects2[0]), ('Class = 0', 'Class = 1'))
    plt.show()

    #To create a proportional bar graph, divide each element of array0 and array1 by the sum

    array0_prop = array0/ np.add(array0,array1)
    array1_prop = array1/ np.add(array0,array1)

    #create a stacked bar graph using these arrays
    ind = np.arange(valDict[plottingVar])
    disp0 = plt.bar(ind, array0_prop,color='b')
    disp1 = plt.bar(ind, array1_prop,bottom = array0_prop,color='r')
    plt.ylabel('Class value proportions')
    plt.title('Class value proportion by '+plottingVar)
    plt.legend( (disp0[0], disp1[0]), ('Class = 0', 'Class = 1'))
    plt.show()


