""" The Restricted Boltzmann Machine

This class implements the restricted boltzmann machine with two layers.
The class should be able to update the states, train and classify """

import numpy as np
from scipy.special import expit
import copy

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
        
        self.b_h = np.random.normal(0., self.params['sigma_b'], self.params['number_hidden'])
        self.b_v = np.random.normal(0., self.params['sigma_b'], self.n_visible + self.n_label)
        self.W = np.random.normal(0., self.params['sigma_W'], (self.n_hidden, self.n_visible + self.n_label))

    def randomStateInit(self):
        """ Initialize the states randomly with equal probability for ones and zeros"""
        
        self.states_h = np.random.randint( 0, 2, len(self.b_h)) 
        self.states_v = np.random.randint( 0, 2, len(self.b_v))

    #def Update_hidden(self):
        



    #def Update(self):
    #    """ Update the complete RBM using Gibbs sampling. The method uses the update hidden units and separately the update visible units. """
    


    def get_states_h(self):
        """ getter hidden states """

        return copy.deepcopy(self.states_h)

    def get_states_v(self):
        """ getter visible states """

        return copy.deepcopy(self.states_v)
