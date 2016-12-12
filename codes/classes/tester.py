""" Tester class

This class tests the predicition of an RBM. The RBM is connected to the tester class and the
used training set is loaded via the dataManager """

from __future__ import print_function
import numpy as np
import sys

class tester( object):
    """ The tester class """

    def __init__(self, RBM = False, DM = False):
        """ Inititate a plain class 

        Keywords: optional -- [ RBM, DM]
            -- RBM: the restricted BM top be connected
            -- DM: the datamanager to be connected
        """

        if RBM:
            self.RBM = RBM
        if DM:
            self.DM = DM

        return

    def connectDataManager(self, DM):
        """ Connect the data Manager to the tester 

        Keywords: [DM]
            -- DM: data manager to be connected
        """

        self.DM = DM

    def connectRBM(self, RBM):
        """ Connect the RBM to the tester 

        Keywords: [RBM]
            -- RBM: RBM to be connected
        """

        self.RBM = RBM

    def setUsedLabels( self, labels):
        """ Set the used labels

        Keywords: [labels]
            -- labels: list of the used labels
        """

        self.labels = labels

    def prepareTesting( self):
        """ Set up the variables for testing """
        
        self.correct = 0
        self.false = 0

        # Classification matrix
        # [i, j] means that the example with label labels[i] was classified as labels[j]
        self.C_mat = np.zeros( (len(self.labels), len(self.labels)) )

    def fullTest( self):
        """ Test the predictions on the complete BM """

        # get the test set from the dataManager
        testSet = self.DM.getTest()
        labelInput = np.zeros(self.RBM.n_label) # There should be no input from the label, i.e. zero visible input for the label neurons

        for i in xrange(len(testSet)):
            
            # Prepare the input
            label = testSet[i,0]
            if not(label in self.labels):
                continue
            feature = testSet[i,1:]/255.
            vinput = feature * 0.95 + 0.025
            vinput = np.log(1./(1./vinput - 1.))
            vinput = np.append( vinput, labelInput)

            # Predict
            prediction = self.RBM.predict( vinput)

            # Evaluate
            if prediction == label:
                self.correct += 1
            else:
                self.false += 1

            self.C_mat[ self.labels.index(label), self.labels.index(prediction)] += 1


            # Report to the console
            j = i + 1
            text = '\rTesting is finished for the %sth test' %j
            print( text, end='')
            sys.stdout.flush()
           
        print('')
        print('Testing has finished')
