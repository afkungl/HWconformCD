import HWconformCD
import numpy as np
import os


print "WARNING: This test only checks if the syntax is working. Sensibility is not checked!"

labels = [1,5,9]

p_RBM = {'number_hidden': 20,
     'number_feature': 784,
     'number_label': 3,
     'sigma_b': 0.5,
     'sigma_W': 0.5}

rbm = HWconformCD.RBM(p_RBM)
rbm.randomInit()
rbm.storeMetaTraining(labels)

if os.path.isfile('../dataManager/mnist_test.csv'):
    sets = '../dataManager/mnist_test.csv'
else:
    sets = 'dataManager/mnist_test.csv'

p_DM = { 'training' : sets ,
      'test' : sets }


DM = HWconformCD.dataManager(p_DM)
DM.loadTest()

p_CD = { 'learning_rate' : 0.001,
         'used_labels' : labels,
         'sigma_w' : 0.5,
         'sigma_b' : 0. }

te = HWconformCD.tester( DM = DM, RBM = rbm)
te.setUsedLabels( labels)
te.prepareTesting()

DM.test = DM.test[:500]

te.fullTest()


test = ( te.correct + te.false == te.C_mat.sum() )

if test:
    print 'tester.py: RBM testing test: PASSED'
else:
    print 'tester.py: RBM testing test: FAILED'
