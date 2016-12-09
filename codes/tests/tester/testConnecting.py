import HWconformCD
import numpy as np
import os


p_RBM = {'number_hidden': 5,
     'number_feature': 12,
     'number_label': 7,
     'sigma_b': 0.5,
     'sigma_W': 0.5}

rbm = HWconformCD.RBM(p_RBM)

if os.path.isfile('mnist_test.csv'):
    sets = '../dataManager/mnist_test.csv'
else:
    sets = 'dataManager/mnist_test.csv'

p_DM = { 'training' : sets ,
      'test' : sets }


DM = HWconformCD.dataManager(p_DM)

te = HWconformCD.tester( DM = DM, RBM = rbm)

test = ( te.DM.test_path == p_DM['test'] ) and (te.RBM.params == p_RBM)

if test:
   print "tester.py: Connection test: PASSED."
else:
   print "tester.py: Connection test: FAILED."
