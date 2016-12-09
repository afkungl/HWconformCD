import HWconformCD
import numpy as np

filename = 'data.yml'

p = {'number_hidden': 4,
     'number_feature': 4,
     'number_label': 3,
     'sigma_b': 0.5,
     'sigma_W': 0.5}

rbm = HWconformCD.RBM(p)

rbm.randomInit()
rbm.saveRBM( filename)

rbm2 = HWconformCD.RBM({}, fromFile = True, filename = filename)

del rbm.visibleInput # Only for testing

t1 = all(rbm2.b_h == rbm.b_h)
del rbm.b_h
t2 = all(rbm2.b_v == rbm.b_v)
del rbm.b_v
t3 = np.all(rbm2.W == rbm.W)
del rbm.W


comp = []
for key in rbm.__dict__:
    #print 'Original: %s' %key
    #print rbm.__dict__[key]
    #print 'Copy: %s' %key
    #print rbm2.__dict__[key]
    #print 'Comparison'
    t = (rbm.__dict__[key] == rbm2.__dict__[key])
    #print t
    comp.append(t)

test = all(comp) and t1 and t2 and t3

if test:
    print 'Load and save RBM with yaml: PASSED'
else:
    print 'Load and save RBM with yaml: FAILED'
