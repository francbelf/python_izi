#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:25:01 2018

@author: francesco
"""
import numpy as np
import time
from izi.izi import izi
import matplotlib.pyplot as plt
plt.ion()

timestart = time.process_time()
#%%

#THIS IS THE TEST data shown in Fig 1 of Blanc et al., 2015
#fluxes from HII region NGC0925 +087 -031 from Table 3 
#of Van Zee et al. 1998, ApJ, 116, 2805


#TOP PANEL (ALL LINES)
flux=[3.44,2.286, 1, 0.553, 0.698, 2.83]
error=[0.17, 0.07,0.05, 0.023, 0.027, 0.15]
id=['oii3726;oii3729','oiii4959;oiii5007', 'hbeta' , 'nii6548;nii6584', \
       'sii6717;sii6731', 'halpha']
   
out2=izi(flux, error, id,\
    intergridfile='interpolgrid_50_50d13_kappa20',
    epsilon=0.1, quiet=False, plot=True)
#%%

#SECOND PANEL from top ([OIII], [OII], Hb)
flux=[3.44,2.286, 1]
error=[0.17, 0.07,0.05]
id=['oii3726;oii3729','oiii4959;oiii5007', 'hbeta' ]
   
out2=izi(flux, error, id,\
    intergridfile='interpolgrid_50_50d13_kappa20',
    epsilon=0.1, quiet=False, plot=True)
#%%
#THIRD PANEL from top ([NII], [OII])
flux=[3.44,0.553]
error=[0.17, 0.023]
id=['oii3726;oii3729', 'nii6548;nii6584']
   
out2=izi(flux, error, id,\
    intergridfile='interpolgrid_50_50d13_kappa20',
    epsilon=0.1, quiet=False, plot=True)

#BOTTOM PANEL ([NII], Ha)
flux=[0.553,  2.83]
error=[0.023,  0.15]
id=[ 'nii6548;nii6584', 'halpha']
   
out2=izi(flux, error, id,\
    intergridfile='interpolgrid_100_100d13_kappa20',
    epsilon=0.1, quiet=False, plot=True)
    
print('Elapsed time: {0} seconds'.format(time.process_time() - timestart))
