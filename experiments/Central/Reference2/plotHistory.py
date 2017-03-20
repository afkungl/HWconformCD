import HWconformCD
import numpy as np
import os


print 'Plotting stuff'

pl = HWconformCD.plotter()
pl.setOutputFolder('plots')

pl.setHistoryResolution( 500)
pl.loadTrainingHistory( 5000)

index = np.arange(100,200)

pl.plotBiasHHist()
pl.plotBiasVHist()
pl.plotWHist( index = index)
