import os
import subprocess

folders = os.listdir('.')

for folder in folders:

    if os.path.isfile(folder):
        continue
    
    fis = os.listdir(folder)

    for f in fis:
        
        if f.endswith('.py'):
            
            print 'Start unittest:............%s' %f
            st = "python %s" % os.path.join(folder,f)
            subprocess.call( st, shell = True)
