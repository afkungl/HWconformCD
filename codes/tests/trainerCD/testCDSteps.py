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

if os.path.isfile('../dataManager/mnist_test.csv'):
    sets = '../dataManager/mnist_test.csv'
else:
    sets = 'dataManager/mnist_test.csv'

p_DM = { 'training' : sets ,
      'test' : sets }


DM = HWconformCD.dataManager(p_DM)


p_CD = { 'learning_rate' : 0.001,
         'used_labels' : labels,
         'sigma_w' : 0.5,
         'sigma_b' : 0. }

trainer = HWconformCD.trainerCD1(p_CD)
trainer.connectRBM(rbm)
trainer.connectDataManager(DM)
trainer.initTraining()
DM.setUsedLabels(labels)
DM.loadTraining()
DM.prepareBag()
miniBatch = DM.getBalancedMiniBatch()

oneexample = miniBatch[0]

# Fist test: one example gives reasonable values
[g_W, g_bh, g_bv] = trainer.getOneGradient( oneexample['feature'], oneexample['label'])
(m,n) = g_W.shape
Nh = len(g_bh)
Nv = len(g_bv)
test1 = (m == 20) and (n == 787) and (Nh == 20) and (Nv == 787)
if test1:
    print 'One example gradient test: PASSED'
else:
    print 'One example gradient test: FAILED'

# Second test: miniBatch learning gives reasonable values
[g_W, g_bh, g_bv] = trainer.getOneMiniBatchGradient( miniBatch)
(m,n) = g_W.shape
Nh = len(g_bh)
Nv = len(g_bv)
test1 = (m == 20) and (n == 787) and (Nh == 20) and (Nv == 787)
if test1:
    print 'One mini batch gradient test: PASSED'
else:
    print 'One mini batch gradient test: FAILED'


