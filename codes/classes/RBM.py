""" The Restricted Boltzmann Machine

This class implements the restricted boltzmann machine with two layers.
The class should be able to update the states, train and classify """

import numpy as np
from scipy.special import expit
import copy
import sys

class RBM(object):
    """ Object of the RBM """


    def __init__ (self, params):
        """ The initialization of the class

        -- Keywords:
            -- params: dictionary of the parameters, which contains
                -- weights: ndarray (number of hidden units * number of visible units)
                -- bias_hidden: ndarray
                -- bias_visible: ndarray
                -- data_class: pointer to the data layer
                -- number_labels: int
                -- number_hidden: int
                -- number_visible: int
                -- sigma_b: real (sigma of the gaussian for the random init for the biases)
                -- sigma_W: real (sigma of the gaussian for the random init for the weights)
        """

        self.params = params

    def randomInit(self):

        self.n_hidden = self.params['number_hidden']
        self.n_visible = self.params['number_visible']
        self.n_label = self.params['number_label']
        self.n_visAll = self.n_visible + self.n_label
        
        # Weights and biases
        self.b_h = np.random.normal(0., self.params['sigma_b'], self.params['number_hidden'])
        self.b_v = np.random.normal(0., self.params['sigma_b'], self.n_visAll)
        self.W = np.random.normal(0., self.params['sigma_W'], (self.n_hidden, self.n_visAll))

        # Visible input (the external input to the visible layer)
        self.visibleInput = np.zeros( self.n_visAll)

    def randomStateInit(self):
        """ Initialize the states randomly with equal probability for ones and zeros"""
        
        self.states_h = np.random.randint( 0, 2, len(self.b_h)) 
        self.states_v = np.random.randint( 0, 2, len(self.b_v))
    
    def setFeature(self, feature):
        """ Set a feature vector onto the visible layer 

        Keywords: [feature]
            -- feature: the feature vector of length self.n_visible to be set to the first instances of the visible layer

        """

        self.states_v[:self.n_visible] = feature

    def getFeature(self):
        """ getter for the actual feature """

        return copy.deepcopy(self.states_v[:self.n_visible])

    def setLabel(self, label):
        """ Set a label vector onto the visible layer

        Keywords: [label]
            -- label: A label vector of length self.n_label

        """

        self.states_v[self.n_visible:] = label

    def getLabel(self):
        """ Getter for the actual label """

        return copy.deepcopy( self.states_v[self.n_visible:])


    def Update_hidden(self):
        """ Update the hidden layer once """
        
        probs = expit(self.W.dot(self.states_v) + self.b_h)
        r = np.random.rand( self.n_hidden )
        self.states_h = np.floor( probs - r + 1.)

    def Update_visible(self, clamped = 'none'):
        """ Update the visible neurons once

        Kewords: clamped
            -- clamped = none: The complete visible layer is updated
            -- clamped = label: The label is clamped
            -- clamped = feature: The feature vector is clamped
        """

        if clamped == 'none':
            probs = expit(self.W.T.dot(self.states_h) + self.b_v + self.visibleInput)
            r = np.random.rand( self.n_visAll )
            self.states_v = np.floor( probs - r + 1.)
        elif clamped == 'label':
            probs = expit(self.W.T[:self.n_visible,:].dot(self.states_h) + self.b_v[:self.n_visible] + self.visibleInput[:self.n_visible])
            r = np.random.rand( self.n_visible )
            self.states_v[:self.n_visible] = np.floor( probs - r + 1.)
        elif clamped == 'feature':
            probs = expit(self.W.T[self.n_visible:,:].dot(self.states_h) + self.b_v[self.n_visible:] + self.visibleInput[self.n_visible:])
            r = np.random.rand( self.n_label )
            self.states_v[self.n_visible:] = np.floor( probs - r + 1.)
        else:
            sys.exit('The variable clamped has an invalid value')



    def Update(self, clamped = 'none'):
        """ Update the complete RBM using Gibbs sampling. The method uses the update hidden units and separately the update visible units. """
    
        self.Update_hidden()
        self.Update_visible(clamped)


    def get_states_h(self):
        """ getter hidden states """

        return copy.deepcopy(self.states_h)

    def get_states_v(self):
        """ getter visible states """

        return copy.deepcopy(self.states_v)
