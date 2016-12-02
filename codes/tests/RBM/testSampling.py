import HWconformCD
import numpy as np



p = {'number_hidden': 1,
     'number_feature': 1,
     'number_label': 0,
     'sigma_b': 0.5,
     'sigma_W': 0.5}

     
rbm = HWconformCD.RBM(p)

print "Sampling test"

rbm.randomInit()
rbm.randomStateInit()

b1 = rbm.b_h[0]
b2 = rbm.b_v[0]
w = rbm.W[0,0]

# Theoretical probs
pTh = np.array([0.,0.,0.,0.])
pTh[0] = np.exp(0.)
pTh[1] = np.exp(b1)
pTh[2] = np.exp(b2)
pTh[3] = np.exp(b1 + b2 + w)
pTh = pTh/np.sum(pTh)

# Sampled distribution
pS = np.array([0.,0.,0.,0.])

for i in xrange(1000000):
    rbm.Update()
    index = 2**0 * rbm.states_h[0] + 2**1 * rbm.states_v[0]
    pS[index] += 1

pS = pS/sum(pS)

print "The theoretical distribution"
print pTh
print "The sampled distribution"
print pS

err = np.abs(pTh - pS)/ pTh

if max(err) < 10**-2.:
    print "Sampling test: PASSED"
else:
    print "Sampling test: FAILED"
