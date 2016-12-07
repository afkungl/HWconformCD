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

if os.path.isfile('mnist_test.csv'):
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
miniBatch = DM.getBalancedMiniBatch()
