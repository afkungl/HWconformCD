import HWconformCD
import numpy as np



p = {'number_hidden': 5,
     'number_feature': 12,
     'number_label': 7,
     'sigma_b': 0.5,
     'sigma_W': 0.5}

rbm = HWconformCD.RBM(p)


# First test
rbm.randomInit()
rbm.randomStateInit()


rbm.setLabel(np.ones(7))

for i in range(10):
    rbm.Update(clamped = 'label')

if np.all(rbm.getLabel() == np.ones(7)):
    print "Test: Update with clamping the labels: PASSED"
else:
    print "Test: Update with clamping the labels: FAILED"



# Second test
rbm.randomStateInit()

rbm.setFeature(np.ones(12))

for i in range(10):
    rbm.Update(clamped = 'feature')

if np.all(rbm.getFeature() == np.ones(12)):
    print "Test: Update with clamping the feature: PASSED"
else:
    print "Test: Update with clamping the feature: FAILED"

