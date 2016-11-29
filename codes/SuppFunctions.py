""" Supplemetary functions module

This module collects the Supplementary functions gahtered in the SuppFunctions folder """

import os
import sys

foF = os.path.join(os.path.dirname(os.path.realpath(__file__)),'SuppFunctions') # The folder of the Functions

sys.path.append( foF)

files = [f for f in os.listdir(foF) if f[-2:]=='py']

for f in files:
    string = 'from %s import *' %f[:-3]
    exec(string)
