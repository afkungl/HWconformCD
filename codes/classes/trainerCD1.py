""" The CD trainer 

The class trains an RBM object with given metaparameters and dataset.
"""

from __future__ import print_function
import numpy as np
import os
import sys
import random
import copy
import time 
import json
import gc

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
        
        self.constrained = False

        self.label_dic = {}
        i = 0
        for l in self.labels:
            rbmLabel = np.zeros(self.n_labels)
            rbmLabel[i] = 1
            self.label_dic[l] = rbmLabel
            i += 1

    def setConstraints(self, w, b):
        """ Set symmetric constraints on the paramters 
        
        Keywords: [w, b]
            -- w: constraint for the weights
            -- b: constraint for the biases
        """

        self.con_w = w
        self.con_b = b
        self.constrained = True

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

    #@profile
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

    #@profile
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
            gr = self.getOneGradient( example['feature'], example['label'])
            a_grW += gr[0]
            a_grbh += gr[1]
            a_grbv += gr[2]

        a_grW = a_grW/m
        a_grbh = a_grbh/m
        a_grbv = a_grbv/m

        return [a_grW, a_grbh, a_grbv]

    #@profile
    def trainRBM( self, N, outFolder = 'trainingData'):
        """ Train the RBM using balanced minibatches  for N steps 
        
        Keywords: N, optinal: outFolder
            -- N : number of training steps, One step is a training step on a minibatch
            -- outFolder: path of the folder to save the data
        """
        
        
        self.DM.setUsedLabels( self.labels)
        self.DM.prepareBag()

        # Prepare outFolder
        if not os.path.exists( outFolder):
            os.makedirs( outFolder)

        # Arrays to save the training history
        self.TH_biash = []
        self.TH_biasv = []
        self.TH_W = []

        # Measure the needed time for training
        t1 = time.clock()

        for i in xrange(N):
           
           # get the gradients
           grad = self.getOneMiniBatchGradient( self.DM.getBalancedMiniBatch())
            
           # Update weights and biases
           self.RBM.W += self.eta_w * grad[0]
           self.RBM.b_h += self.eta_bh * grad[1]
           self.RBM.b_v += self.eta_bv * grad[2]
           if self.constrained:
              self.RBM.W.clip( -self.con_w, self.con_w, out = self.RBM.W)
              self.RBM.b_h.clip( -self.con_b, self.con_b, out = self.RBM.b_h)
              self.RBM.b_v.clip( -self.con_b, self.con_b, out = self.RBM.b_v)

           # Save the weights and biases
           j = i + 1
           if i%500 == 0:
              W = 'weights%08d.npy' %i
              b_h = 'biasH%08d.npy' %i
              b_v = 'biasV%08d.npy' %i
              np.save( os.path.join( outFolder, W), self.RBM.W)
              np.save( os.path.join( outFolder, b_h), self.RBM.b_h)
              np.save( os.path.join( outFolder, b_v), self.RBM.b_v)
              
              if i%5000 == 0:
                gc.collect()


           # Report to the console
           
           text = '\rTraining is finished for the %sth training step' %j
           print( text, end='')
           sys.stdout.flush()

        dt = time.clock() - t1
           
        self.RBM.storeMetaTraining( self.labels)
        print('')
        print('Training has finished')
        time_rep = 'The training took: %s sec' %dt
        print(time_rep)


### Classic CD1 trainer ###

class trainerCD1Classic( trainerCD1):

    def getOneGradient(self, feature, label):
        """ Do CD on one example of the batch and return the obtained gradient
            The clamping is done in the classic manner. I.e. the pictue is binarized adn clamped

        Keywords: [feature, label]
           -- feature: the feature vector
           -- label: name of the label

        Return: [gradient_w, gradient_bh, gradient_bv]
           -- gradient_w: Matrix of the weigth gradient
           -- gradient_bh: Vector of the bias gradient for the hidden units
           -- gradient_bv: Vector of the bias gradient for the visible units
         """
        
        self.RBM.randomStateInit()

        # First create the visible input from the data
        visibleClamped = np.append(feature, self.label_dic[label])
        r = np.ones( self.RBM.n_visAll ) * 0.5
        visibleClamped = np.floor( visibleClamped - r + 1.)
        self.RBM.states_v = visibleClamped
        v_data = visibleClamped

        # Initialize in a random state and do the updates

        h_data = self.RBM.Update_hidden()

        v_recon = self.RBM.Update_visible()
        h_recon = self.RBM.Update_hidden()

        # Calculate the necessary gradients
        gradient_w = np.outer(h_data,v_data) - np.outer(h_recon, v_recon)
        gradient_bh = h_data - h_recon
        gradient_bv = v_data - v_recon
 
        return [ gradient_w, gradient_bh, gradient_bv]


class trainerCD1Unsupervized( trainerCD1):  
    """ Class which trains the BM in an unsupervized manner """
    
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
        vinput = feature * 0.95 + 0.025
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


class trainerCD1UnsupervizedClassic( trainerCD1):  
    """ Class which trains the BM in an unsupervized manner """
    
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

        self.RBM.randomStateInit()

        # First create the visible input from the data
        visibleClamped = feature
        r = np.ones( self.RBM.n_visAll ) * 0.5
        visibleClamped = np.floor( visibleClamped - r + 1.)
        self.RBM.states_v = visibleClamped
        v_data = visibleClamped


        # Initialize in a random state and do the updates
        

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
