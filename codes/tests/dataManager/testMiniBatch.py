import HWconformCD
import numpy as np
import os

if os.path.isfile('mnist_test.csv'):
    sets = 'mnist_test.csv'
else:
    sets = 'dataManager/mnist_test.csv'

p = { 'training' : sets ,
      'test' : sets }
labels = [1,4,7,8]

DM = HWconformCD.dataManager(p)

DM.loadTraining()
DM.prepareBag()
DM.setUsedLabels(labels)

miniB = DM.getBalancedMiniBatch()

p_a = []
for sample in miniB:
    if sample['label'] in labels:
        p_a.append(True)
    else:
        p_a.append(False)

if len(miniB) == len(labels):
    p_a.append(True)
else:
    p_a.append(False)

passed = np.all(p_a)

if passed:
    print 'Balanced mini batch test: PASSED'
else:
    print 'Balanced mini batch test: FAILED'
