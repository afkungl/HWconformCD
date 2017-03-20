import HWconformCD
import numpy as np
import os


print 'testing the trained RBM'

RBM_file = 'trainedRBM.yml'

# Load the RBM
rbm = HWconformCD.RBM({}, fromFile = True, filename = RBM_file)

# Set up the data Manager and load the test set
p_DM = { 'training' : '' ,
      'test' : '../../../data/mnist_test.csv' }

DM = HWconformCD.dataManager(p_DM)
DM.loadTest()
#DM.test = DM.test[:3000]

# Set up the tester
te = HWconformCD.tester( DM = DM, RBM = rbm)
te.setUsedLabels( rbm.labels)
te.prepareTesting()
te.fullTest()

cl_rate = float(te.correct) / float(te.correct + te.false)

print "The measured classification rate is %s" %cl_rate

print 'The mixing matrix is:'
print te.C_mat

# Save the mixing matrix
if not os.path.exists('TestData'):
	os.makedirs('TestData')

np.savetxt('TestData/Mixing.txt', te.C_mat)

np.savetxt('TestData/ClassRate.txt', np.array([cl_rate]))