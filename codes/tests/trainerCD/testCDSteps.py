import HWconformCD
import numpy as np
import os

labels = [1,5,9]

p_RBM = {'number_hidden': 5,
     'number_feature': 12,
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
[g_W, g_bh, g_vh] = trainer.getOneGradient( oneexample['feature'], oneexample['label'])
(m,n) = g_W.shape
Nh = len(g_bh)
Nv = len(g_bv)
test1 = (m == 5) and (n == 15) and (Nh == 5) and (Nv == 15)
if test1:
    print 'One example gradient test: PASSED'
else:
    print 'One example gradient test: FAILED'

# Second test: miniBatch learning gives reasonable values
[g_W, g_bh, g_vh] = trainer.getOneMiniBatchGradient( oneexample['feature'], oneexample['label'])
(m,n) = g_W.shape
Nh = len(g_bh)
Nv = len(g_bv)
test1 = (m == 5) and (n == 15) and (Nh == 5) and (Nv == 15)
if test1:
    print 'One mini batch gradient test: PASSED'
else:
    print 'One mini batch gradient test: FAILED'


