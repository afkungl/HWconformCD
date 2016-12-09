import os
import subprocess
import sys

folders = os.listdir('.')

for folder in folders:

    if os.path.isfile(folder):
        continue
    
    fis = os.listdir(folder)

    for f in fis:
        
        if f.endswith('.py'):
        
            print '=================================='
            print 'Start unittest:............%s' %f
            sys.stdout.flush()
            st = "python %s" % os.path.join(folder,f)
            subprocess.call( st, shell = True)
