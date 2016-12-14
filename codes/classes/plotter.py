import matplotlib
matplotlib.use('Agg')

"""
Plotter

Class to automatically plot the data """




import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import os
import numpy as np
from scipy.special import expit

# Always use this!
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)


class plotter( object):
    
    def __init__(self):
        """
        Empty Constructor
        """


        #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        #rc('text', usetex=True)

        self.outFolder = '.'

    def setOutputFolder( self, folder):
        """ Setter for the output folder 
        
        Keywords: [folder]
            -- folder: path of the output folder
        """

        self.outFolder = folder

        if not os.path.exists( self.outFolder):
            os.makedirs( self.outFolder)

    # Connect the RBM to the plotter

    def connectRBM( self, RBM):
        """ Connect the RBM to the Plotter
    
        Keywords: RBM
            -- RBM: the RBM ro be connected
        """
        
        self.RBM = RBM

    # Plotter part fot plotting the training

    def loadTrainingHistory( self, N, inFolder = 'trainingData'):
        """ Load the training history from the yaml database

        Keywords: N, optinal: inFolder
            -- inFolder: path of the fodler containing the training data
            -- N: The number of used training steps
        """
        
        W_hist = []
        bh_hist = []
        bv_hist = []
        for i in np.arange( 0, N-1, 500):
            W = 'weights%08d.npy' %i
            b_h = 'biasH%08d.npy' %i
            b_v = 'biasV%08d.npy' %i
            W_hist.append(np.load( os.path.join( inFolder, W)).flatten())
            bh_hist.append(np.load( os.path.join( inFolder, b_h)))
            bv_hist.append(np.load( os.path.join( inFolder, b_v)))

        self.W_hist = np.array(W_hist)
        self.bh_hist = np.array(bh_hist)
        self.bv_hist = np.array(bv_hist)

    def plotBiasHHist( self, name = 'biasHHistory.png', index = False):
        """ Plot the bias (hidden) history as function of the training steps

        Keywords: optional: [name, index]
            -- name of the output file
            -- index: Specify the index of the biases to be plotted. Bt default all of them are plotted
        """

        # x array
        N = np.array(range( len( self.bh_hist[:,1]))) * 500

        f, ax = plt.subplots(1)

        if not index:
            loop_over = range(self.bh_hist.shape[1])
        else:
            loop_over = index

        for i in loop_over:
            ax.plot( N, self.bh_hist[:,i])

        ax.set_xlabel('Number of training steps')
        ax.set_ylabel('Bias (hidden)')
        plotTo = os.path.join( self.outFolder, name)
        print plotTo
        plt.savefig( plotTo)
        plt.clf()

    def plotBiasVHist( self, name = 'biasVHistory.png', index = False):
        """ Plot the bias (visible) history as function of the training steps

        Keywords: optional: [name, index]
            -- name of the output file
            -- index: Specify the index of the biases to be plotted. Bt default all of them are plotted
        """

        # x array
        N = np.array(range( len( self.bv_hist[:,1]))) * 100
        
        f, ax = plt.subplots(1)

        if not index:
            loop_over = range(self.bv_hist.shape[1])
        else:
            loop_over = index

        for i in loop_over:
            ax.plot( N, self.bv_hist[:,i])

        ax.set_xlabel('Number of training steps')
        ax.set_ylabel('Bias (visible)')
        plotTo = os.path.join( self.outFolder, name)
        print plotTo
        plt.savefig( plotTo, dpi = 150)
        plt.clf()


    def plotWHist( self, name = 'WHistory.png', index = False):
        """ Plot the bias (visible) history as function of the training steps

        Keywords: optional: [name, index]
            -- name of the output file
            -- index: Specify the index of the biases to be plotted. Bt default all of them are plotted
        """

        # x array
        N = np.array(range( len( self.W_hist[:,1]))) * 100
        
        f, ax = plt.subplots(1)

        if not index:
            loop_over = range(self.W_hist.shape[1])
        else:
            loop_over = index

        for i in loop_over:
            ax.plot( N, self.W_hist[:,i])

        ax.set_xlabel('Number of training steps')
        ax.set_ylabel('Weights')
        plotTo = os.path.join( self.outFolder, name)
        print plotTo
        plt.savefig( plotTo, dpi = 150)
        plt.clf()

    def visualizeDream( self, N, inFolder = 'dreamData', outFolder = 'dreamData', picSize = (28, 28)):
        """ Visualize the dream produced by the RBM. The function uses the same name convention as RBM.dream()

        Keywords: [ N] optional: [ inFolder, outFolder]
            -- N: the number of dream steps to be visualized
           optional:
            -- inFolder: Folder containing the data
            -- outFolder: output folder for the produced video """


        #FFMpegWriter = manimation.writers['ffmpeg']
        #metadata = dict(title='Movie Test', artist='Matplotlib',
        #        comment='Movie support!')
        #writer = FFMpegWriter(fps=15, metadata=metadata)

        if not os.path.exists( outFolder):
            os.makedirs( outFolder)

        # Set up the figure
        fig, ax = plt.subplots(1)
        
        # Initalize the animation
        i = 1
        feature = 'dreamFeature%08d.npy' %i
        picture = np.load( os.path.join( inFolder, feature)).reshape( picSize)
        ThePic = ax.imshow(picture, cmap = cm.gray_r, interpolation = 'nearest')

        def init():
            return

        def animate(n):
            feature = 'dreamFeature%08d.npy' %n
            picture = np.load( os.path.join( inFolder, feature)).reshape( picSize)
            ThePic.set_data( picture)
            return

        ani = FuncAnimation(fig, animate, init_func=init, frames = range(N), interval = 100., blit = False, repeat = True)

        video_name = os.path.join( outFolder, 'Dream.mp4')
        ani.save( video_name)


    def visualizeDreamCond( self, N, inFolder = 'dreamData', outFolder = 'dreamData', picSize = (28, 28)):
        """ Visualize the dream produced by the RBM. The function uses the same name convention as RBM.dream()
            The pictures are infered from the hidden layer probabilities.

        Keywords: [ N] optional: [ inFolder, outFolder]
            -- N: the number of dream steps to be visualized
           optional:
            -- inFolder: Folder containing the data
            -- outFolder: output folder for the produced video """


        #FFMpegWriter = manimation.writers['ffmpeg']
        #metadata = dict(title='Movie Test', artist='Matplotlib',
        #        comment='Movie support!')
        #writer = FFMpegWriter(fps=15, metadata=metadata)

        if not os.path.exists( outFolder):
            os.makedirs( outFolder)

        # Set up the figure
        fig, ax = plt.subplots(1)
        
        # Initalize the animation
        i = 1
        hidden = 'dreamHidden%08d.npy' %i
        hidden_state = np.load( os.path.join( inFolder, hidden))
        probs = expit(self.RBM.W.T.dot(hidden_state) + self.RBM.b_v)[:self.RBM.n_feature]
        picture = probs.reshape( picSize)
        ThePic = ax.imshow(picture, cmap = cm.gray_r, interpolation = 'nearest')

        def init():
            return

        def animate(n):
            hidden = 'dreamHidden%08d.npy' %n
            hidden_state = np.load( os.path.join( inFolder, hidden))
            probs = expit(self.RBM.W.T.dot(hidden_state) + self.RBM.b_v)[:self.RBM.n_feature]
            picture = probs.reshape( picSize)
            ThePic.set_data( picture)
            return

        ani = FuncAnimation(fig, animate, init_func=init, frames = range(N), interval = 100., blit = False, repeat = True)

        video_name = os.path.join( outFolder, 'DreamCond.mp4')
        ani.save( video_name)

    # Plot functions for the RBM

    def plotWHistogram( self, name = 'WHist.png'):
        """ Plot the histogram of the weights and save it
        
        Keywords: optional: name
            -- name: name of the picture
        """

        W = self.RBM.W.flatten()

        plt.hist( W, 80)
        plt.xlabel( 'Weight value')
        plt.ylabel( 'Frequency [1]')
        plt.title( 'Histogram of the weights')
        plt.grid(True)

        plt.savefig( os.path.join( self.outFolder, name))
        plt.clf()


    def plotBHHistogram( self, name = 'bhHist.png'):
        """ Plot the histogram of the hidden biases and save it
        
        Keywords: optional: name
            -- name: name of the picture
        """

        plt.hist( self.RBM.b_h, 80)
        plt.xlabel( 'Bias value')
        plt.ylabel( 'Frequency [1]')
        plt.title( 'Histogram of the hidden biases')
        plt.grid(True)

        plt.savefig( os.path.join( self.outFolder, name))
        plt.clf()

    def plotBVHistogram( self, name = 'bvHist.png'):
        """ Plot the histogram of the hidden biases and save it
        
        Keywords: optional: name
            -- name: name of the picture
        """

        plt.hist( self.RBM.b_v, 80)
        plt.xlabel( 'Bias value')
        plt.ylabel( 'Frequency [1]')
        plt.title( 'Histogram of the visible biases')
        plt.grid( True)

        plt.savefig( os.path.join( self.outFolder, name))
        plt.clf()

    def plotHistograms( self):
        """ Plot the histograms for the weights and the biases """

        self.plotWHistogram()
        self.plotBVHistogram()
        self.plotBHHistogram()
