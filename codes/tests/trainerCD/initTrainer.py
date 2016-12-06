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


p_CD = { 'learning_rate' : 0.001,
         'used_labels' : [1,2,3],
         'sigma_w' : 0.5,
         'sigma_b' : 0. }

trainer = HWconformCD.trainerCD1(p_CD)
trainer.connectRBM(rbm)
trainer.connectDataManager(DM)
trainer.initTraining()


test1 = p_RBM == trainer.RBM.params
if test1:
    print "RBM connection: PASSED"
else:
    print "RBM connection: FAILED"
test2 = p_DM['training'] == trainer.DM.training_path
if test2:
    print "DM connection: PASSED"
else:
    print "DM connection: FAILED"
test3 = np.all(np.array(trainer.RBM.b_h) == 0.)
if test3:
    print "Initailisation: PASSED"
else:
    print "Initialisation: FAILED"
