""" The Restricted Boltzmann Machine

This class implements the restricted boltzmann machine with two layers.
The class should be able to update the states, train and classify """

import numpy as np
from scipy.special import expit
import copy
import sys
import json
import os

class RBM(object):
    """ Object of the RBM """


    def __init__ (self, params, fromFile = False, filename = ''):
        """ The initialization of the class

        -- Keywords:
            -- params: dictionary of the parameters, which contains
                -- weights: ndarray (number of hidden units * number of visible units)
                -- bias_hidden: ndarray
                -- bias_visible: ndarray
                -- data_class: pointer to the data layer
                -- number_labels: int
                -- number_hidden: int
                -- number_feature: int
                -- sigma_b: real (sigma of the gaussian for the random init for the biases)
                -- sigma_W: real (sigma of the gaussian for the random init for the weights)
        """

        # Option to initialize the RBM from a file
        if fromFile:
            self.loadRBM( filename)
            return

        self.params = params
        self.trained = False
        self.labels = []

    def randomInit(self):

        self.n_hidden = self.params['number_hidden']
        self.n_feature = self.params['number_feature']
        self.n_label = self.params['number_label']
        self.n_visAll = self.n_feature + self.n_label

        
        # Weights and biases
        if not self.params['sigma_b'] == 0.:
            self.b_h = np.random.normal(0., self.params['sigma_b'], self.params['number_hidden'])
            self.b_v = np.random.normal(0., self.params['sigma_b'], self.n_visAll)
        else:
            self.b_h = np.zeros( self.params['number_hidden'])
            self.b_v = np.zeros( self.n_visAll)
        if not self.params['sigma_W'] == 0.:
            self.W = np.random.normal(0., self.params['sigma_W'], (self.n_hidden, self.n_visAll))
        else:
            self.W = np.zeros((self.n_hidden, self.n_visAll))


        # Visible input (the external input to the visible layer)
        self.visibleInput = np.zeros( self.n_visAll)

    def randomStateInit(self):
        """ Initialize the states randomly with equal probability for ones and zeros"""
        
        self.states_h = np.random.randint( 0, 2, len(self.b_h))
        self.states_v = np.random.randint( 0, 2, len(self.b_v))
    
    def setFeature(self, feature):
        """ Set a feature vector onto the visible layer 

        Keywords: [feature]
            -- feature: the feature vector of length self.n_feature to be set to the first instances of the visible layer

        """

        self.states_v[:self.n_feature] = feature

    def setVisibleInput(self, vinput):
        
        self.visibleInput = vinput
    
    def delVisibleInput(self):
        """ set the visible Input to zero """

        self.visibleInput = np.zeros( self.n_visAll)

    def getFeature(self):
        """ getter for the actual feature """

        return copy.deepcopy(self.states_v[:self.n_feature])

    def setLabel(self, label):
        """ Set a label vector onto the visible layer

        Keywords: [label]
            -- label: A label vector of length self.n_label

        """

        self.states_v[self.n_feature:] = label

    def getLabel(self):
        """ Getter for the actual label """

        return copy.deepcopy( self.states_v[self.n_feature:])

    #@profile
    def Update_hidden(self):
        """ Update the hidden layer once 
        
        Returns: [states_h]
            -- states_h: the updated hidden states
        """
        
        probs = expit(self.W.dot(self.states_v) + self.b_h)
        #print probs[:11]
        r = np.random.rand( self.n_hidden )
        self.states_h = np.floor( probs - r + 1.)

        return copy.deepcopy(self.states_h)

    #@profile
    def Update_visible(self, clamped = 'none'):
        """ Update the visible neurons once

        Kewords: [clamped]
            -- clamped = none: The complete visible layer is updated
            -- clamped = label: The label is clamped
            -- clamped = feature: The feature vector is clamped
        
        Returns: [visible_states]
            -- visible_states: Returns the visible states
        """

        if clamped == 'none':
            probs = expit(self.W.T.dot(self.states_h) + self.b_v + self.visibleInput)
            r = np.random.rand( self.n_visAll )
            self.states_v = np.floor( probs - r + 1.)
        elif clamped == 'label':
            probs = expit(self.W.T[:self.n_feature,:].dot(self.states_h) + self.b_v[:self.n_feature] + self.visibleInput[:self.n_feature])
            r = np.random.rand( self.n_feature )
            self.states_v[:self.n_feature] = np.floor( probs - r + 1.)
        elif clamped == 'feature':
            probs = expit(self.W.T[self.n_feature:,:].dot(self.states_h) + self.b_v[self.n_feature:] + self.visibleInput[self.n_feature:])
            r = np.random.rand( self.n_label )
            self.states_v[self.n_feature:] = np.floor( probs - r + 1.)
        else:
            sys.exit('The variable clamped has an invalid value')

        return copy.deepcopy(self.states_v)

    def saveBMExplicit(self):

        # Weights
        N = self.n_hidden + self.n_visAll
        W = np.zeros((N,N))
        W[self.n_visAll:,:self.n_label] = self.W[:,self.n_feature:]
        W[self.n_visAll:,self.n_label:self.n_visAll] = self.W[:,:self.n_feature]
        W = W + W.T
        np.savetxt('W.txt', W)

        # biases
        b = np.zeros(N)
        b[:self.n_label] = self.b_v[self.n_feature:]
        b[self.n_label:self.n_visAll] = self.b_v[:self.n_feature]
        b[self.n_visAll:] = self.b_h
        np.savetxt('b.txt', b)

    def loadRBMExplicit(self, wFile, bFile):
        """ The counterpart of the method save RBM explicit """

        # Null the current parameters
        self.W = self.W * 0.
        self.b_v = self.b_v * 0.
        self.b_h = self.b_h * 0.

        # Weights
        W = np.loadtxt(wFile)
        self.W[:,self.n_feature:] = W[self.n_visAll:,:self.n_label]
        self.W[:,:self.n_feature] = W[self.n_visAll:,self.n_label:self.n_visAll]

        # biases
        b = np.loadtxt(bFile)
        self.b_v[self.n_feature:] = b[:self.n_label]
        self.b_v[:self.n_feature] = b[self.n_label:self.n_visAll] 
        self.b_h = b[self.n_visAll:] 


    #@profile
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

    def storeMetaTraining(self, labels):
        """ Report that the RBM has been trained by a trainer

        Keywords: [labels]
            -- labels: list of the stored labels
        """

        self.trained = True
        self.labels = labels

    ############################################
    ## Writing and saving the RBM into a json file

    def saveRBM(self, filename):
        """
        The methods dumps the RBM into a json file

        Keywords: [ filename]
            -- filename: path for the file to save
        """
       
        dictToSave = { 'hidden' : self.n_hidden,
                       'visible' : self.n_visAll,
                       'feature' : self.n_feature,
                       'label' : self.n_label,
                       'trained' : self.trained,
                       'labels' : self.labels,
                       'params': self.params,
                       'W' : self.W.tolist(),
                       'bias_h' : self.b_h.tolist(),
                       'bias_v' : self.b_v.tolist() }
        
    
        with open( filename, 'w') as outfile:
            json.dump(dictToSave, outfile, indent=4, sort_keys=True)

    def loadRBM( self, filename):
        """
        The method loads an rbm from the specified json database

        Keywords: [filename]
            -- filename: path of the json file
        """

        with open( filename, 'r') as infile:
            Dict = json.load( infile)

        self.n_hidden = Dict['hidden']
        self.n_visAll = Dict['visible']
        self.n_feature = Dict['feature']
        self.n_label = Dict['label']
        self.trained = Dict['trained']
        self.labels = Dict['labels']
        self.params = Dict['params'] 
        self.W = np.array(Dict['W'])
        self.b_h = np.array(Dict['bias_h'])
        self.b_v = np.array(Dict['bias_v'])

    ##############################

    def predict( self, visIn):
        """ Predict the label using the RBM given the input in the visible layer

        Keywords: [visIn]
            -- visIn: Input to be clamped to the visible input

        Returns: [ label]
            -- the predicted label
        """

        self.setVisibleInput( visIn)
        self.randomStateInit()

        # Burn in
        for n in xrange(30):
            self.Update()

        # Actual Sampling
        pred_arr = np.zeros( self.n_label)
        for n in xrange(200):
            self.Update()
            pred_arr += self.states_v[self.n_feature:]
        

        prediction = self.labels[pred_arr.argmax()]

        return prediction

    def predictClassic( self, feature):
        """ Predict the label using the RBM given the input in the visible layer

        Keywords: [visible]
            -- visible: Input to be clamped to the visible layer

        Returns: [ label]
            -- the predicted label
        """

        self.randomStateInit()
        self.delVisibleInput()
        self.states_v[:self.n_feature] = feature
        
        # Burn in
        for n in xrange(30):
            self.Update( clamped = 'feature')

        # Actual Sampling
        pred_arr = np.zeros( self.n_label)
        for n in xrange(200):
            self.Update( clamped = 'feature')
            pred_arr += self.states_v[self.n_feature:]
        

        prediction = self.labels[pred_arr.argmax()]

        return prediction

    def dream( self, N, outFolder = 'dreamData'):
        """ Let the RBM dream freely

        Keywords: [N] optional: outFolder
            -- N: Number of dreaming steps
            -- outFolder: The path of the folder to dumb the states
        """
        # Prepare for dreaming
        self.delVisibleInput()
        self.randomStateInit()
        self.states_h = np.ones( self.n_hidden)
        self.states_v = np.zeros( self.n_visAll)
        if not os.path.exists( outFolder):
            os.makedirs( outFolder)

        # Burn in
        #for i in xrange(100):
        #    self.Update()

        # Dream and save
        for i in range( N):
            
            feature = 'dreamFeature%08d.npy' %i
            label = 'dreamLabel%08d.npy' %i
            hidden = 'dreamHidden%08d.npy' %i
            np.save( os.path.join( outFolder, feature), self.states_v[:self.n_feature])
            np.save( os.path.join( outFolder, label), self.states_v[self.n_feature:])
            np.save( os.path.join( outFolder, hidden), self.states_h)       
            self.Update()
            
    def dreamLabel( self, N, label, outFolder = 'dreamData'):
        """ Let the RBM dream freely

        Keywords: [N] optional: outFolder
            -- N: Number of dreaming steps
            -- outFolder: The path of the folder to dumb the states
        """
        # Prepare for dreaming
        feature = np.zeros( self.n_feature)
        labelI = np.zeros( self.n_label)
        labelI[ self.labels.index( label)] = 1.
        vinput = np.append(feature, labelI) * 0.95 + 0.025
        vinput = np.log(1./(1./vinput - 1.))
        self.setVisibleInput(vinput)
        self.randomStateInit()
        self.states_h = np.ones( self.n_hidden)
        if not os.path.exists( outFolder):
            os.makedirs( outFolder)

        # Burn in
        for i in xrange(100):
            self.Update()

        # Dream and save
        for i in range( N):
            self.Update()
            feature = 'dreamFeature%08d.npy' %i
            label = 'dreamLabel%08d.npy' %i
            hidden = 'dreamHidden%08d.npy' %i
            np.save( os.path.join( outFolder, feature), self.states_v[:self.n_feature])
            np.save( os.path.join( outFolder, label), self.states_v[self.n_feature:])
            np.save( os.path.join( outFolder, hidden), self.states_h)


    def dreamLabelClassic( self, N, label, outFolder = 'dreamData'):
        """ Let the RBM dream freely

        Keywords: [N] optional: outFolder
            -- N: Number of dreaming steps
            -- outFolder: The path of the folder to dumb the states
        """
        # Prepare for dreaming
        labelI = np.zeros( self.n_label)
        labelI[ self.labels.index( label)] = 1.
        self.delVisibleInput()
        self.randomStateInit()
        self.states_h = np.ones( self.n_hidden)
        if not os.path.exists( outFolder):
            os.makedirs( outFolder)

        self.states_v[self.n_feature:] = labelI

        # Burn in
        for i in xrange(100):
            self.Update( clamped = 'label')

        # Dream and save
        for i in range( N):
            self.Update( clamped = 'label')
            feature = 'dreamFeature%08d.npy' %i
            label = 'dreamLabel%08d.npy' %i
            hidden = 'dreamHidden%08d.npy' %i
            np.save( os.path.join( outFolder, feature), self.states_v[:self.n_feature])
            np.save( os.path.join( outFolder, label), self.states_v[self.n_feature:])
            np.save( os.path.join( outFolder, hidden), self.states_h)


    def reportStatistic( self, filename):
        """ Gather and report the statistics of the weights and biases and save them into a file

            Keywords:
                -- filename: name of the target file
        """

        # Initialize the dictionary
        stat = {}

        # Gather stat for the hidden bias
        b_h = {}
        b_h['mean'] = self.b_h.mean()
        b_h['std'] = self.b_h.std()
        b_h['max'] = self.b_h.max()
        b_h['min'] = self.b_h.min()
        stat['b_h'] = b_h

        # Gather stat for the visible bias
        b_v = {}
        b_v['mean'] = self.b_v.mean()
        b_v['std'] = self.b_v.std()
        b_v['max'] = self.b_v.max()
        b_v['min'] = self.b_v.min()
        stat['b_v'] = b_v

        # gather statistics for the Weights
        W = {}
        W['mean'] = self.W.mean()
        W['std'] = self.W.std()
        W['max'] = self.W.max()
        W['min'] = self.W.min()
        stat['W'] = W

        # Report to the console
        for key in stat:
            print key
            for key2 in stat[key]:
                print '%s is %s' %(key2, stat[key][key2])

        with open( filename, 'w') as outfile:
            json.dump(stat, outfile, indent=4, sort_keys=True)