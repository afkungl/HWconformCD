""" The CD trainer 

The class traines an RBM object with given metaparameters and dataset.
"""

import numpy as np
import os
import sys
import random

class trainerCD1(object):

    def __init__(self, params):
        """ Constructor method

        Keywords [ params]:
        -- params: dictionary of the required parameters
            -- learning_rate: the used learning rate if contant
            -- used_labels: vector of the used labels 
                -- first label in the list gets the vector [1,0,0,0,...]
                   second label gets [0,1,0,0...] etc.
            -- sigma_w: initial sigma for the weights NOTE: overrides the RBM sigma
            -- sigma_b: initial sigma for the biases NOTE: overrides the RBM sigma """
            

        self.eta = params['learning_rate']
        self.labels = params['used_labels']
        self.n_labels = len(self.labels)
        self.sigma_b = params['sigma_b']
        self.sigma_w = params['sigma_w']

    def connectRBM(self, RBM):
        """ Connect the RBM object to the trainer

        Keywords [RBM]:
            -- RBM - the RBm object """

        self.RBM = RBM

    def connectDataManager(self, DM):
        """ Connect the DM object to the trainer

        Keywords [ DM]:
            -- DM - the dataManager object """

        self.DM = DM
    
    def initTraining(self):
        """ Initialize the weigths and biases of the BM """

        self.RBM.params['sigma_b'] = self.sigma_b
        self.RBM.params['sigma_W'] = self.sigma_w

        self.RBM.randomInit()
