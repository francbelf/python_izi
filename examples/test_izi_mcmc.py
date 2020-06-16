#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:25:01 2018

@author: francesco
"""
import numpy as np
import time
from izi.izi_MCMC import izi_MCMC
import matplotlib.pyplot as plt


timestart = time.process_time()


flux=[3.44,2.286, 1, 0.553, 0.698, 2.83]
error=[0.17, 0.07,0.05, 0.023, 0.027, 0.15]
id=['oii3726;oii3729','oiii4959;oiii5007', 'hbeta' , 'nii6548;nii6584', \
       'sii6717;sii6731', 'halpha']

out2=izi_MCMC(flux, error, id,\
     intergridfile='interpolgrid_100_100d13_kappaINF', epsilon=0.1, quiet=False, 
     plot=True,plot_z_steps=False,plot_q_steps=False,
     logqprior=None)
     
# now imposing a Gaussian prior on the ionisation parameter
#%%
out2=izi_MCMC(flux, error, id,
    intergridfile='interpolgrid_100_100d13_kappaINF', epsilon=0.1, quiet=False, 
    plot=False,logqprior=[7.1,0.5],plot_q_steps=True)
#%%
print('Elapsed time: {0} seconds'.format(time.clock() - timestart))

import corner
samples = out2.samples
ndim=3

# This is the empirical mean of the sample:
value0 = np.median(samples, axis=0)
valueup = np.percentile(samples,  66, axis=0)
valuedown = np.percentile(samples, 34, axis=0)
print(value0, valueup, valuedown)

# Make the base corner plot
figure = corner.corner(samples, \
                labels=[r"$\rm 12+log(O/H)$", r"$\rm q \ [cm^{-2}]$",  \
                        r"$ \rm E(B-V)  \ [mag]$"],\
                show_titles=True,
                title_kwargs={"fontsize": 10},\
                label_kwargs={"fontsize": 14},\
                data_kwargs={"ms": 0.6})

# Extract the axes

axes = np.array(figure.axes).reshape((ndim, ndim))

# Loop over the diagonal
for i in range(ndim):
    ax = axes[i, i]

    ax.axvline(value0[i], color="r")
    ax.axvline(valueup[i], color="r", ls='--')
    ax.axvline(valuedown[i], color="r", ls='--')
#    ax.axhline(value1[i], color="b")
    
plt.show()
