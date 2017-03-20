import HWconformCD
import numpy as np
import os
import sys
import copy

def main():
    sys.stdout.flush()
    
    # Numbers to be learned
    labels = [0,1,2,3,4,5,6,7,8,9]
    

    # THE PRETRAINING
    print 'Working on the pretraining:'

    # Set up the RBM and initialize randomly
    prep_RBM = {'number_hidden': 500,
         'number_feature': 784,
         'number_label': 0,
         'sigma_b': 0.05,
         'sigma_W': 0.05}
    
    prerbm = HWconformCD.RBM(prep_RBM)
    prerbm.randomInit()
    
    
    # Set up the data Manager and load the data
    p_DM = { 'training' : '../../../data/mnist_train.csv' ,
          'test' : '' }
    
    DM = HWconformCD.dataManager(p_DM)
    DM.loadTraining()
    
    # Set up the trainer and initialize the training
    p_CD = { 'learning_rate' : 0.001,
             'used_labels' : labels,
             'sigma_w' : 0.005,
             'sigma_b' : 0.0 }
    
    pretrainer = HWconformCD.trainerCD1Unsupervized(p_CD)
    pretrainer.connectRBM(prerbm)
    pretrainer.connectDataManager(DM)
    pretrainer.setLearningRates( 0.005, 0.005, 0.005)
    pretrainer.initTraining()
    #pretrainer.setConstraints( 1000., 1000.)
    
    
    # Do the training and save the resulting RBM
    pretrainer.trainRBM(300000)

    # The normal training
    # Set up the RBM and initialize randomly plus the data from the pretrained RBM
    p_RBM = {'number_hidden': 500,
         'number_feature': 784,
         'number_label': 10,
         'sigma_b': 0.00,
         'sigma_W': 0.05}
    
    rbm = HWconformCD.RBM(p_RBM)
    rbm.randomInit()
    rbm.W[:,:rbm.n_feature] = prerbm.W
    rbm.b_h = prerbm.b_h
    rbm.b_v[:rbm.n_feature] = prerbm.b_v

    # Continue training with the supervised learning
    # Set up the trainer and initialize the training
    p_CD = { 'learning_rate' : 0.001,
             'used_labels' : labels,
             'sigma_w' : 0.005,
             'sigma_b' : 0.0 }
    
    trainer = HWconformCD.trainerCD1Classic(p_CD)
    trainer.connectRBM(rbm)
    trainer.connectDataManager(DM)
    trainer.setLearningRates( 0.002, 0.002, 0.002)
    #trainer.setConstraints( 100.5, 100.5)
    
    
    # Do the training and save the resulting RBM
    trainer.trainRBM(300000)

    rbm.saveRBM('trainedRBM.yml')

if __name__=="__main__":
    main()


