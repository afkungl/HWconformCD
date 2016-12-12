""" The Data manager layer 

This layer loads the required training and test data.
It prepares the mini-batches for the training.
"""

import numpy as np
import os
import sys
import random

class dataManager(object):

    def __init__(self, params):
        """ Constructor method

        Keywords:
        -- params: dictionary of the required parameters
            -- training: path of the training set
            -- test: path of the test set """
        
        self.training_path = params['training']
        self.test_path = params['test']
        np.random.seed(12345678)

    def loadTest(self):
        """ Load the test set into the memory """

        if not os.path.isfile(self.test_path):
            sys.exit('ERROR: Invalid path for test set')
        else:
            self.test = np.loadtxt( self.test_path, delimiter = ',', dtype = np.int32)

    def loadTraining(self):
        """ Load the training set into the memory """

        if not os.path.isfile(self.training_path):
            sys.exit('ERROR: Invalid path for training set')
        else:
            self.training = np.loadtxt( self.training_path, delimiter = ',', dtype = np.int32)

    def prepareBag(self):
        """ Prepare a bag for all the available trainingsamples
        
        A bag of available training samples is prepared in terms of indexes """

        self.bagsize = len(self.training)
        self.bag = range(self.bagsize)

    def setUsedLabels(self, labels):
        """ Set the labels which are used for training or testing """

        self.labels = labels
        self.n_labels = len(self.labels)

    def getBalancedMiniBatch(self):
        """ Return a Balanced mini batch, i.e. one sample of each class

        Returns: [ batch]
            -- batch: a list of dictinaries of length self.n_labels
                  in the dictionary
                  -- label : label 
                  -- feature : the feature vector
        """
        batch = []
        gathered = []

        for l in self.labels:
            
            accepted = False

            while not accepted:
            
                proposed = random.sample(self.bag,1)[0]
                l_proposed = self.training[proposed,0]

                if l_proposed == l and (not (proposed in gathered)):
                    batch.append({'label': l_proposed, 'feature': self.training[proposed,1:]/255.})
                    gathered.append(proposed)
                    accepted = True

        return batch

    def getTest( self):
        """ getter for the test set. It provides a pointer to the test set. """

        return self.test
         

