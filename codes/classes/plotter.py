"""
Plotter

Class to automatically plot the data """


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import yaml

# Always use this!
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


class plotter( object):
    
    def __init__(self):
        """
        Empty Constructor
        """


        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)

        self.outFolder = '.'

    def setOutputFolder( self, folder):
        """ Setter for the output folder 
        
        Keywords: [folder]
            -- folder: path of the output folder
        """

        self.outFolder = folder

        if not os.path.exists( self.outFolder):
            os.makedirs( self.outFolder)

    # Plotter part fot plotting the training

    def loadTrainingHistory( self, filename):
        """ Load the training history from the yaml database

        Keywords: [filename]
            -- filename: path of the yaml database
        """

        with open( filename, 'r') as infile:
            Dict = yaml.load( infile)

        self.W_hist = Dict['W']
        self.bh_hist = Dict['b_h']
        self.bv_hist = Dict['b_v']

    def plotBiasHHist( self, name = 'biasHHistory.png'):
        """ Plot the bias (hidden) history as function of the training steps

        Keywords: optional: [name]
            -- name of the output file
        """

        # x array
        N = np.array(range( len( self.bh_hist[:,1]))) * 100

        f, ax = plt.subplots(1)


        for i in range(self.bh_hist.shape[1]):
            ax.plot( N, self.bh_hist[:,i])

        ax.set_xlabel('Number of training steps')
        ax.set_ylabel('Bias (hidden)')
        plotTo = os.path.join( self.outFolder, name)
        print plotTo
        plt.savefig( plotTo, dpi = 150)
        plt.clf()

    def plotBiasVHist( self, name = 'biasVHistory.png'):
        """ Plot the bias (visible) history as function of the training steps

        Keywords: optional: [name]
            -- name of the output file
        """

        # x array
        N = np.array(range( len( self.bv_hist[:,1]))) * 100
        
        f, ax = plt.subplots(1)


        for i in range(self.bv_hist.shape[1]):
            ax.plot( N, self.bv_hist[:,i])

        ax.set_xlabel('Number of training steps')
        ax.set_ylabel('Bias (visible)')
        plotTo = os.path.join( self.outFolder, name)
        print plotTo
        plt.savefig( plotTo, dpi = 150)
        plt.clf()


    def plotWHist( self, name = 'WHistory.png'):
        """ Plot the bias (visible) history as function of the training steps

        Keywords: optional: [name]
            -- name of the output file
        """

        # x array
        N = np.array(range( len( self.W_hist[:,1]))) * 100
        
        f, ax = plt.subplots(1)


        for i in range(self.W_hist.shape[1]):
            ax.plot( N, self.W_hist[:,i])

        ax.set_xlabel('Number of training steps')
        ax.set_ylabel('Weights')
        plotTo = os.path.join( self.outFolder, name)
        print plotTo
        plt.savefig( plotTo, dpi = 150)
        plt.clf()
