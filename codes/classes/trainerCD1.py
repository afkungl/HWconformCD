""" The CD trainer 

The class trains an RBM object with given metaparameters and dataset.
"""

from __future__ import print_function
import numpy as np
import os
import sys
import random
#import time # Only for debugging

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
        self.eta_w = self.eta
        self.eta_bv = self.eta
        self.eta_bh = self.eta
        self.labels = params['used_labels']
        self.n_labels = len(self.labels)
        self.sigma_b = params['sigma_b']
        self.sigma_w = params['sigma_w']

        self.label_dic = {}
        i = 0
        for l in self.labels:
            rbmLabel = np.zeros(self.n_labels)
            rbmLabel[i] = 1
            self.label_dic[l] = rbmLabel
            i += 1

    def setLearningRates( self, eta_w, eta_bv, eta_bh):
        """ Set the learining rates for the trainer. Especially the learning rates for the weights and biases can be independently modified.
        If this method is not used then there is only one learning rate for all the parameters.

        Keywords: [ eta_w, eta_bv, eta_bh]
            -- eta_w: weights learning rate
            -- eat_bv: visible bias learning rate
            -- eta_bh: hidden learning rate
        """

        self.eta_w = eta_w
        self.eta_bv = eta_bv
        self.eta_bh = eta_bh

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

    def getOneGradient(self, feature, label):
        """ Do CD on one example of the batch and return the obtained gradient

        Keywords: [feature, label]
           -- feature: the feature vector
           -- label: name of the label

        Return: [gradient_w, gradient_bh, gradient_bv]
           -- gradient_w: Matrix of the weigth gradient
           -- gradient_bh: Vector of the bias gradient for the hidden units
           -- gradient_bv: Vector of the bias gradient for the visible units
         """

        # First create the visible input from the data
        vinput = np.append(feature, self.label_dic[label]) * 0.95 + 0.025
        vinput = np.log(1./(1./vinput - 1.))
        self.RBM.setVisibleInput(vinput)

        # Initialize in a random state and do the updates
        self.RBM.randomStateInit()

        v_data = self.RBM.Update_visible()
        h_data = self.RBM.Update_hidden()

        self.RBM.delVisibleInput()

        v_recon = self.RBM.Update_visible()
        h_recon = self.RBM.Update_hidden()

        # Calculate the necessary gradients
        gradient_w = np.outer(h_data,v_data) - np.outer(h_recon, v_recon)
        gradient_bh = h_data - h_recon
        gradient_bv = v_data - v_recon
 
        return [ gradient_w, gradient_bh, gradient_bv]

    def getOneMiniBatchGradient( self, miniBatch):
        """ Take one minibatch as it is created by the data manager and get the average gradient over the minibatch
        
        Keywords: [miniBatch]
            -- miniBatch: as created by the data manager

        Returns: [a_grW, a_grh, a_grv]
            -- a_grW: average gradient of the weights
            -- a_grbh: average gradient of the hidden biases
            -- a_grbv: average gradient of the visible biases
        """

        m = len(miniBatch)
        a_grW = np.zeros( (self.RBM.n_hidden, self.RBM.n_visAll))
        a_grbh = np.zeros( self.RBM.n_hidden)
        a_grbv = np.zeros( self.RBM.n_visAll)

        for example in miniBatch:
            [grW, grbh, grbv] = self.getOneGradient( example['feature'], example['label'])
            a_grW = grW
            a_grbh = grbh
            a_grbv = grbv

        a_grW = a_grW/m
        a_grbh = a_grbh/m
        a_grbv = a_grbv/m

        return [a_grW, a_grbh, a_grbv]

    def trainRBM( self, N):
        """ Train the RBM using balanced minibatches  for N steps """
        
        self.DM.setUsedLabels( self.labels)
        self.DM.loadTraining()
        self.DM.prepareBag()

        for i in xrange(N):
           
           # get the gradients
           [ grad_W, grad_bh, grad_bv] = self.getOneMiniBatchGradient( self.DM.getBalancedMiniBatch())
            
           # Update weights and biases
           self.RBM.W = self.RBM.W + self.eta_w * grad_W
           self.RBM.b_h = self.RBM.b_h + self.eta_bh * grad_bh
           self.RBM.b_v = self.RBM.b_v + self.eta_bv * grad_bv
           
           # Report to the console
           j = i + 1
           text = '\rTraining is finished for the %sth training step' %j
           print( text, end='')
           sys.stdout.flush()
           
        self.RBM.storeMetaTraining( self.labels)
        print('')
        print('Training has finished')





