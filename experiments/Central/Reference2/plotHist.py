import HWconformCD
import numpy as np
import os

RBM_file = 'trainedRBM.yml'

# Load the RBM
rbm = HWconformCD.RBM({}, fromFile = True, filename = RBM_file)

print 'Plotting stuff'

pl = HWconformCD.plotter()
pl.setOutputFolder('plots')
pl.connectRBM( rbm)

pl.plotHistograms()
pl.plotReceptiveFields( outFolder = 'plots/receptive')

rbm.reportStatistic('stat.txt')