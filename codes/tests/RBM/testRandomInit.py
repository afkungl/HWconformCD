import HWconformCD
import numpy as np



p = {'number_hidden': 4,
     'number_feature': 12,
     'number_label': 3,
     'sigma_b': 0.5,
     'sigma_W': 0.5}

rbm = HWconformCD.RBM(p)

rbm.randomInit()
rbm.randomStateInit()

if len(rbm.get_states_h()) == 4 and len(rbm.get_states_v()) == 15 and rbm.W.shape == (4,15):
    print 'Random initialization test: PASSED'
else:
    print 'Random initialization test: FAILED'
