import HWconformCD
import numpy as np
import os

if os.path.isfile('mnist_test.csv'):
    sets = '../dataManager/mnist_test.csv'
else:
    sets = 'dataManager/mnist_test.csv'

p = { 'training' : sets ,
      'test' : sets }


DM = HWconformCD.dataManager(p)

DM.loadTest()
DM.loadTraining()

if len(DM.test) == 10000:
    print 'Loading test set: PASSED'
else:
    print 'Loading test set: FAILED'


if len(DM.training) == 10000:
    print 'Loading training set: PASSED'
else:
    print 'Loading training set: FAILED'
