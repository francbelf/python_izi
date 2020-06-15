#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:13:01 2017
+
; NAME:
;
;   izi_mcmc
;
; PURPOSE:
;
;   Compute the posterior PDF of the gas-phase metallicity, ionization 
;   parameter and dust EBV given a set of observed emission line fluxes
;   and errors, and a photo-ionization model grid. The code assumed a Calzetti 
;   extinction law.
;   
; CALLING SEQUENCE:
;
;  result = IZI(flux, error, id, gridfile=None, templ_dir='', grid=None,
                 plot=True, plot_z_steps=False, plot_q_steps=False, plot_ebv_steps=False, 
                 plot_derivatives=False, logohsun=None, epsilon=None, nz=None, nq=None, 
                 intergridfile=None, integrid=None, outgridfile=False, logzlimits=None, 
                 logqlimits=None, logzprior=None, logqprior=None, 
                 nonorm=0, quiet=False)
;
;
; INPUT PARAMETERS:
;
;    FLUX: array of emission line fluxes in arbitrary units. Fluxes
;    are nornalized internally in the code to the flux of H-beta, or
;    in the case H-beta is not provided to the flux of the brightest
;    input emission line. Fluxes should NOT be corrected for dust extinction.
;
;    ERROR: array of emission line flux errors. Upper limits can be
;    provided by setting the a value of -666 in the FLUX array and
;    providing the 1 sigma upper limit in the ERROR parameter.
;
;    ID: array of strings containing the emission line IDs. The
;    names of the lines must comply with the IDs stored in the FITS
;    table storing the photo-ionization model grid. For all the grids that
;    are provided with IZI the following are some (not all) of the
;    adopted IDs:
;
;          'oii3726'    for [OII]-3726
;          'oii3729'    for [OII]-3729
;          'hgamma'     for H_gamma-4341
;          'oiii4959'   for [OIII]-4959
;          'hbeta'      for H_beta-4861
;          'oiii5007'   for [OIII]-5007
;          'oi6300'     for [OI]-6300
;          'nii6548'    for [NII]-6548
;          'halpha'     for H_alpha-6563
;          'nii6584'    for [NII]-6584
;          'sii6717'    for [SII]-6717
;          'sii6731'    for [SII]-6731 
;          'siii9068'   for [SIII]-9068
;          'siii9532'   for [SIII]-9532 
;
;
;    For a complete list of lines included in a particular grid read
;    the photo-ionization model grid into Python and check
;    the ID tag. 
;
;    Summed doublets (e.g. [OII]3726,9 at low resolution) can be
;    considered as a single line by providing the IDs of the
;    components separated by a semicolon in IDIN (e.g. 'oii3726;oii3729').
;    Note that this should only be done when the doublet are close in 
;    wavelength since the code applies an extinction correction based on the
;    wavelength of the first line
;
;        
; KEYWORD PARAMETERS:
;
;    GRIDFILE: string containing the filename for the photo-ionization
;    model grid to be used. If not provided IZI defaults to the
;    Levesque et al. 2010 models with high mass loss, constant star
;    formation history, n=1e2 cm^-3 and age of 6.0Myr. The format of
;    the grid files is a FITS table read as an astropy table with an
;    entry for each (Z,q) element in the grid and the following tags for
;    each entry:
;      
;          NAME            STRING    name of the photo-ionization model
;          LOGOHSUN        FLOAT     assumed 12+log(O/H) solar abundance in the model 
;          LOGZ            FLOAT     log of metallicity (Z) in solar units
;          LOGQ            FLOAT     log of ion. parameter in log(cm/s) units
;          ID              STRING[Nlines]    emission line IDs
;          FLUX            FLOAT[Nlines]     emission line fluxes
;
;    PLOT: set this keyword to True to produce corner plots of the
;    parameters PDFs.
;
;    PLOT_Z_STEPS: set this keyword to True to produce a plot of the log(Z) values found by
;    the walkers of the mcmc as a function of the number of steps (use to check for convergence).
;
;    PLOT_Q_STEPS: set this keyword to True to produce a plot of the log(q) values found by
;    the walkers of the mcmc as a function of the number of steps (use to check for convergence).
;
;    PLOT_EBV_STEPS: set this keyword to True to produce a plot of the E(B-V) values found by
;    the walkers of the mcmc as a function of the number of steps (use to check for convergence).
;   
;    LOGOHSUN: set this keyword to a user supplied value which is used
;    instead of the LOGOHSUN value in the model grid file.
;
;    EPSILON: systematic uncertainty in dex for the emission line
;    fluxes in the model (see equation 3 in Blanc et al. 2014). If not
;    provided the code assumes a default value is 0.1 dex.
;
;    NZ: number of log(Z) elements in the interpolated grid, if not
;    provided the default is NZ=100. NZ must be 100 or larger to minimise the
;    error in the interpolation.
;   
;    NQ: number of log(q) elements in the interpolated grid, if not
;    provided the default is NZ=50. NQ must be 100 or larger to minimise the
;    error in the interpolation.
;
;    INTERGRIDFILE: string containing the filename for an already
;    interpolated photo-ionization model grid. If provided this file
;    is used intead of GRIDFILE and the interpolation step is
;    skiped. This interpolated grid file should be created using the
;    OUTGRIDFILE keyword. It is strongly adivces to use this keyword for
;    speeding computation time when running IZI_MCMC for a large sample of objects.
;
;    OUTGRIDFILE: set this keyword to True to produce to save the
;    interpolated grid file for latter use with INTERGRIDFILE
;
;    LOGZLIMITS: 2 element array containing the lower and upper
;    log(Z) limits for the interpolated grid (section 3 of Blanc et
;    al. 2014) in units of 12+log(O/H)
;
;    LOGQLIMITS: 2 element array containing the lower and upper
;    log(Z) limits for the interpolated grid (section 3 of Blanc et
;    al. 2014) in units of log(cm/s)
;
;    NONORM: set this keyword to avoid the normalization of the line
;    fluxes. This is useful when using model grids of line ratios
;    instead of line fluxes.
;
;    QUIET: set to true to avoid printing output to terminal
; 
;    LOGQPRIOR: 2 element list containing the expected value of the parameter logq
;    and its uncertainty. A gaussian centred on this value multiplies the
;    likelihood function.
;
;    LOGZPRIOR: 2 element list containing the expected value of the parameter logZ
;    and its uncertainty. A gaussian centred on this value multiplies the
;    likelihood function.
;
; OUTPUT:
;
;    RESULT.sol: a dictionary containing the output best fit parameters and PDFs
;
;      id: names of the emission lines in the photo-ionization model
;      flux: nomalized input line fluxes, same size as id
;      error: error in normalized line fluxes, same size as id
;      Z, err_down_Z, err_up_Z:  median, 16th and 84th percentiles of the metallicity PDF in units of 12+log(O/H)
;      q, err_down_q, err_up_q: median, 16th and 84th percentiles of the ionization parameter PDF in units of log(cm/s)
;      ebv, err_down_ebv, err_up_ebv: median, 16th and 84th percentiles of the dust attenuation E(B_V) in units of mag
;      zarr, qarr, ebvarr: array of metallicity, ionization parameter, extinction values in units of 12+log(O/H), log(cm/s), mag
;      z_pdf, q_pdf, ebv_pdf: 1D marginalized metallicity PDF as a function of zarr, qarr, ebvarr
;      chi2: chi^2 between observed fluxes and model fluxes at mode of the joint PDF
;      acc_fraction: acceptance fraction
;      flag: [znpeaks, qnpeaks, ebvnpeaks, zlimit, qlimit, ebvlimit] with
;           znpeaks, qnpeaks, ebvnpeaks: number of peaks in log(Z), log(q) and E(B-V) marginalized PDFs;
;           zlimit, qlimit, ebvlimit: flags stating if the marginalized PDF is bound (0), or is either an upper (1) or lower (2) limit, or completely unconstrained (3).
;      Z_max, q_max, ebv_max: array with 12+log(O/H), log(q), E(B-V) max values and the corrisponding PDF values, with a length equal to zarr, qarr, ebvarr length
;      .fig: The figure object corresponding to the corner plots of the output .samples: the full MCMC samples array. It has dimensions (Nsamples, 3),
             where 3 corresponds to the 3 free paramters: metallicity (12+log(O/H), ionisation parameter (logq) and EBV (mag).
;
;
;    RESULT.line_stat: a dictionary containing the best model fit for each line and 
;      other diagnostic information 
;          
;      id: names of the emission lines, same size and *same order* as the input flux array
;      fobs: normalised emission line flux, same size and *same order* as the input flux array
;      eobs: normalised emission line flux error, same size and *same order* as the input flux array
;      fmod: normalised best fit model flux, same size and *same order* as the input flux array
;      emod: normalised best fit model flux error, same size and *same order* as the input flux array
;      chi2_line: chi2 for each emission line, same size and *same order* as the input flux array
;
; USAGE EXAMPLE:
;   see test_mcmc.py
;
; MODIFICATION HISTORY: dates are in European format
;
;    V1.0 - Created by G.A. Blanc
;    V2.0 - Translated into Python by F. Belfiore, 23 Jan 2018line
;    V2.1 - (25/03/18 FB) Final tweaks to the Python version and updated the docstring. 
;    V3.1.0 - (04/06/18 FB) First version of izi_MCMC
;    V3.1.1 - (23/08/18 FB) Added functionality for having a different epsilon for the 
;       Balmer lines (only HA implemented), in order to avoid biases in the extinction.
;       Also added samples as output.
;    V3.1.2 - (23/08/18 FB): Added modelFlux object to calculate the model fluxes
;       associated with each MCMC sample and the intrinsic Balmer decrement PDF
;    V4.0 - Added possibility of inserting a Gaussian prior on log(q)
;           and/or 12+log(O/H) variables by M. Mingozzi, 1 March 2019
;           (see Mingozzi et. al. A&A 636, A42 2020 for more details)
;    V5.0.0 - (9/06/20 MM): Upgraded to Python3
;    V5.0.1 - (10/06/20 FB): checked to release
; =====================================================================================

@author: francesco belfiore
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
import astropy.table as table
from scipy import interpolate
from scipy.integrate import simps
from scipy import special
from scipy.signal import find_peaks
import pdb
from os import path
import time
import emcee
import corner
from emcee.autocorr import integrated_time
import random
import warnings
warnings.filterwarnings("ignore")

from izi import fb_extinction as ext
import izi as izi_package

class modelFluxes(object):
    def __init__(self, izi_obj):
        self.samples=izi_obj.samples
        self.grid = izi_obj.intergrid
        self.logohsun=izi_obj.logohsun
        model_fluxes = np.zeros( (self.samples.shape[0], \
                                  len(izi_obj.intergrid['ID'][0])) )
        for ii in range(self.samples.shape[0]):
            out1= grid.value_grid_at_no_interp(self.grid, \
                        self.samples[ii,0] -self.logohsun ,self.samples[ii,1])
            model_fluxes[ii, :] = out1
        self.model_fluxes = model_fluxes

    def calculate_instrinsic_Balmer_decrements(self):
        
        wha = self.grid['ID'][0]=='halpha'
        return self.model_fluxes[:,wha]
    
class grid(object):
    
    def __init__(self, gridfile, templ_dir=None,logohsun=None):
    
        self.gridfile=gridfile
        self.logohsun=logohsun
        self.templ_dir=templ_dir
        
        if self.templ_dir == None:
            self.templ_dir='/grids/'
            print('looking for grids in the izi/grids directory...')

        # READ GRID      
        try:
            grid0=table.Table.read(self.templ_dir+self.gridfile+'.fits')
            grid0.convert_bytestring_to_unicode()
        except IOError:
            raise IOError('no grid file found in '+self.templ_dir+self.gridfile+'.fits')
            
        # get rid of the empty spaces around line names
        grid0id= [num.strip() for num in grid0['ID'][0]]
        #        number of lines
        nlines0=len(grid0['ID'][0])
        #  number of steps in log(Z) * number of steps in log(q)
        ngrid0=len(grid0['LOGZ'])
        #        rename grid0["ID"] to get rid of the empty spaces
        grid0['ID']=[grid0id]*ngrid0
        
        #pdb.set_trace()
        
        self.grid0=grid0

        # TAKE SOLAR OXYGEN ABUNDANCE FROM MODEL GRID IF NOT PROVIDED        
        if self.logohsun ==None:
            self.logohsun = grid0[0]['LOGOHSUN']


    def apply_limits(self, logzlimits=None, logqlimits=None):
        # CUT GRID TO LOGZLIMITS AND LOGQLIMITS 
        self.logzlimits=logzlimits
        self.logqlimits=logqlimits
        
        if self.logzlimits==None:
             self.logzlimits=[np.min(self.grid0['LOGZ']+self.logohsun), 
                   np.max(self.grid0['LOGZ']+self.logohsun)]
        if self.logqlimits==None:
            self.logqlimits=[np.min(self.grid0['LOGQ']), 
                   np.max(self.grid0['LOGQ'])]
       
        self.grid0=self.grid0[ (self.grid0['LOGZ']+self.logohsun >= self.logzlimits[0]) & 
                (self.grid0['LOGZ']+self.logohsun <= self.logzlimits[1]) &
                (self.grid0['LOGQ'] >= self.logqlimits[0]) & 
                (self.grid0['LOGQ'] <= self.logqlimits[1]) ]
        return self
    
    def interpolate_grid(self, nz=50, nq=50):
        
        self.nz=nz
        self.nq=nq
        zarr=np.linspace(np.min(self.grid0['LOGZ']), np.max(self.grid0['LOGZ']), self.nz)
        qarr=np.linspace(np.min(self.grid0['LOGQ']), np.max(self.grid0['LOGQ']), self.nq)
        
        nlines0=len(self.grid0['ID'][0])
        fluxarr=np.zeros((self.nz, self.nq, nlines0))  
        grid_x, grid_y = np.meshgrid(zarr, qarr)
        intergrid=self.nz*self.nq
#            define the new interpolated grid as a table 
        intergrid=table.Table()
        intergrid['LOGQ']=grid_y.flatten()
        intergrid['LOGZ']=grid_x.flatten()
        intergrid['LOGOHSUN']=[self.grid0['LOGOHSUN'][0]]*self.nq*self.nz
        intergrid['ID']=[self.grid0['ID'][0]]*self.nq*self.nz
        
        flux=np.array(self.grid0['FLUX'])
        logzin=np.array(self.grid0['LOGZ'])
        logqin=np.array(self.grid0['LOGQ'])

        for i in range(nlines0): 
             fluxarr[:,:,i]=interpolate.griddata( (logzin,logqin),
                    flux[:,i], (grid_x, grid_y), method='cubic')
           
        #  GOING FROM A 2D grid to a 1D grid
        intergrid['FLUX']= self.make_grid_1d(intergrid, grid_x, grid_y, fluxarr) 
        
        self.intergrid=intergrid
        return self

    @staticmethod
    def make_grid_1d(intergrid, grid_x, grid_y, fluxarr):
        nintergrid=len(intergrid['LOGZ'])
        nlines0=len(intergrid['ID'][0])
        intergrid_flux= np.zeros((nintergrid, nlines0))     
        for j in range(nlines0):     
            for i in range(nintergrid):
                ww= (grid_x == intergrid['LOGZ'][i]) & (grid_y == intergrid['LOGQ'][i])
                flux2d=fluxarr[:,:,j]
                intergrid_flux[i,j]=flux2d[ww]   
        return intergrid_flux
    
    def value_grid_at(self, zz, qq):
        flux=np.array(self.grid0['FLUX'])
        logzin=np.array(self.grid0['LOGZ'])
        logqin=np.array(self.grid0['LOGQ'])
        nlines0=len(self.grid0['ID'][0])
        flux_at=np.zeros(nlines0)

        for i in range(nlines0): 
             flux_at[i]=interpolate.griddata( (logzin,logqin),
                    flux[:,i], (zz, qq), method='linear')
        return flux_at
    
    @staticmethod
    def value_grid_at_no_interp(grid0, zz, qq):
        flux=np.array(grid0['FLUX'])
        logzin=np.array(grid0['LOGZ'])
        logqin=np.array(grid0['LOGQ'])
        www = np.argmin( (logzin-zz)**2 + (logqin-qq)**2 )
        flux_at=flux[www]
        return flux_at

class izi_MCMC(object):
    plt.ioff()
    
    def __init__(self, flux, error, id, gridfile=None, templ_dir=None,
                 logzlimits=None, logqlimits=None, logohsun=None,
                 epsilon=0.1, nz=100, nq=100,
                 intergridfile=None, outgridfile=False, 
                 nonorm=0,
                 quiet=False, plot=True, plot_z_steps=False, plot_q_steps=False, plot_ebv_steps=False, plot_derivatives=False,
                 logqprior=None, logzprior=None):


        #DECLARE INPUT TO SELF
        self.flux = flux          #  flux array
        self.error = error       # error flux array
        self.id = np.copy(id)  # IDs of different emission lines
        self.gridfile = gridfile
        self.templ_dir = templ_dir
        self.logzlimits = logzlimits
        self.logqlimits = logqlimits
        self.intergridfile=intergridfile
        self.nonorm=nonorm
        self.outgridfile=outgridfile
        self.logohsun=logohsun
        self.plot=plot
        self.plot_z_steps=plot_z_steps
        self.plot_q_steps= plot_q_steps
        self.plot_ebv_steps=plot_ebv_steps
        self.plot_derivatives=plot_derivatives
        self.nz=nz
        self.nq=nq
        self.quiet=quiet
        self.epsilon=epsilon
        self.logzprior=logzprior
        self.logqprior=logqprior
        
        nlines_in=len(self.flux)
        assert len(self.error) == nlines_in and len(self.id) == nlines_in, \
        'ERROR Flux, Error, and ID arrays do not have the same number of elements'
        
        assert self.nz>99 and self.nq>99,\
        'ERROR, nz and nq must be larger than 100 for proper interpolation of the model grid'
#

        # INPUT FILES CHECKING    
        # IF NOT SPECIFIED BY USER USE DEFAULT Levesque models with density 
        # 10^2 cm-3, composite SF, and 6Myr age          
        if self.gridfile == None:
            self.gridfile='l09_high_csf_n1e2_6.0Myr'
            #  self.gridfile='d13_kappa20'
        else:
            self.gridfile=gridfile
        if self.templ_dir==None:
            self.templ_dir = path.dirname(path.realpath(izi_package.__file__))[:-4]+'/grids/'
        
        
        if self.intergridfile == None:
            # PREPARE ORIGINAL GRID
            
            # READ GRID using the grid class
            grid0=grid(self.gridfile, templ_dir=self.templ_dir)
            
            # APPLY LIMITS to grid        
            grid0.apply_limits(logzlimits=self.logzlimits, logqlimits=self.logqlimits)
            #     
            if self.logohsun is None:
                self.logohsun=grid0.logohsun
                
            nlines0=len(grid0.grid0['ID'][0])
            #  number of steps in log(Z) * number of steps in log(q)
                
            #INTERPOLATE GRID
#            pdb.set_trace()
            grid0.interpolate_grid(nz=self.nz, nq=self.nq)
            self.intergrid=grid0.intergrid
        
            #DEFINE PARAMTERS OF GRID
#            zarr=np.linspace(np.min(self.intergrid['LOGZ']), np.max(self.intergrid['LOGZ']), self.nz)
#            qarr=np.linspace(np.min(self.intergrid['LOGQ']), np.max(self.intergrid['LOGQ']), self.nq)
#            nintergrid=len(self.intergrid['ID'])
             
            # Read emission line wavelengths and match them to the current order
            # of adopted Cloudy grid. Wavelengths needed for extinction
    
            line_params=ascii.read(path.dirname(path.realpath(izi_package.__file__))+'/line_names.txt') 
            line_wav=np.zeros(nlines0)  
            
            for ii in  range(nlines0):
                line_name=self.intergrid['ID'][0][ii]
                ww = (line_params['line_name']==line_name)
                
                assert ww.sum() == 1, 'ERROR: ===== Line ID '+\
                  self.intergrid['ID'][0][ii]+'not included in wavelength list====='
                    
                line_wav[ii]=line_params['wav'][ww]
            grid0.intergrid['WAV']=[line_wav]*self.nq*self.nz
            
            self.intergrid=grid0.intergrid
            
             # WRITE INTERPOLATED GRID IF USER WANTS TO
            if self.outgridfile ==True:
                a=self.intergrid
                a.write(self.templ_dir+'/interpolgrid_'+str(self.nz)+'_'+\
                        str(self.nq)+self.gridfile+'.fits', overwrite=True) 
        else:
            # READ GRID using the grid class
            grid0=grid(self.intergridfile, templ_dir=self.templ_dir)
            self.intergrid=grid0.grid0
                    
            nlines0=len(self.intergrid['ID'][0])
                            
            self.nz=len(np.unique(self.intergrid['LOGZ']))
            self.nq=len(np.unique(self.intergrid['LOGQ']))
  
            if self.logohsun is None:
                self.logohsun=grid0.logohsun
            
            # Read emission line wavelengths and match them to the current order
            # of adopted Cloudy grid. Wavelengths needed for extinction
    
            line_params=ascii.read(path.dirname(path.realpath(__file__))+'/line_names.txt') 
            line_wav=np.zeros(nlines0)  
          
            #pdb.set_trace()
            for ii in range(nlines0):
                line_name=self.intergrid['ID'][0][ii]
                ww = (line_params['line_name']==line_name)
                assert ww.sum() == 1, 'ERROR: ===== Line ID '+\
                  self.intergrid['ID'][0][ii]+'not included in wavelength list====='
                    
                line_wav[ii]=line_params['wav'][ww]
            grid0.grid0['WAV']=[line_wav]*self.nq*self.nz
            self.intergrid=grid0.grid0
 
        # Check for summed sets of lines in input ID array and sum fluxes in grid
        #  All fluxes are summed to the first line and ID is set to that line 
        for i in range(nlines_in):
            idsum=self.id[i].split(';')
            if len(idsum) >1:    
                for j in range(len(idsum)-1):
                   w0= (self.intergrid['ID'][0] == idsum[0])
                   wj= (self.intergrid['ID'][0] == idsum[j+1])
                   self.intergrid['FLUX'][:,w0]=self.intergrid['FLUX'][:,w0] +\
                    self.intergrid['FLUX'][:,wj]
            self.id[i]=idsum[0]
      
        #; CREATE DATA STRUCTURE CONTAINING LINE FLUXES AND ESTIMATED PARAMETERS
        dd={'id': self.intergrid['ID'][0],       # line id
           'flux': np.zeros(nlines0)+np.nan,      # line flux      
           'error': np.zeros(nlines0)+np.nan,
           'flag': np.zeros(6),
           'Z_max': np.zeros((2,19))+np.nan,
           'q_max': np.zeros((2,19))+np.nan,
           'ebv_max': np.zeros((2,19))+np.nan}
                 
        #FILL STRUCTURE WITH LINE FLUXES
        for i in range(nlines_in):
              auxind=(dd['id'] == self.id[i])
              nmatch=auxind.sum()
              assert nmatch == 1, 'ERROR: ===== Line ID '+self.id[i]+'not recognized ====='
              dd['flux'][auxind]=self.flux[i]
              dd['error'][auxind]=self.error[i]              
        
        #  INDEX LINES WITH MEASUREMENTS
        good=(np.isfinite(dd['error']))
#        ngood=good.sum()
        measured=(np.isfinite(dd['flux']))
        upperlim=((np.isfinite(dd['error'])) & (dd['flux'] == -666))
        
        
        flag0=np.zeros(nlines0)
        flag0[measured]=1      #measured flux
        flag0[upperlim]=2      #upper limit on flux
        #This array has length ngood, which is the number of lines with 
        #given error measurements. If error is given but no flux this is treated
        #as an upper limit
        flag=flag0[good]
        
        # NORMALIZE LINE FLUXES TO HBETA OR
        # IF ABSENT NORMALIZE TO BRIGHTEST LINE

        if self.nonorm ==0: #; use nonorm for line ratio fitting
            idnorm='hbeta'
            in_idnorm= (dd['id']==idnorm)
            
            if (np.isnan(dd['flux'][in_idnorm]) | (dd['flux'][in_idnorm] ==-666)):
                a=dd['id'][measured]
                idnorm=a[[np.argmax(dd['flux'][measured])]]
                in_idnorm= (dd['id']==idnorm)
            
            norm=dd['flux'][in_idnorm]

        #NORMALISE INPUT FLUXES
            
            dd['flux'][measured]=dd['flux'][measured]/norm[0]
            dd['error'][good]=dd['error'][good]/norm[0]
            
            dd['flux'][upperlim] = -666
        
        dd['epsilon']=np.zeros(len(dd['flux'])) + self.epsilon*np.log(10)
        
        in_idnorm= (dd['id']=='halpha')
        dd['epsilon'][in_idnorm]=0.01*np.log(10)

        # pdb.set_trace()
       
        #Define zrange and qrange for use in the prior calculation    
        zrange=[np.min(self.intergrid['LOGZ']), np.max(self.intergrid['LOGZ'])]
        qrange=[np.min(self.intergrid['LOGQ']), np.max(self.intergrid['LOGQ'])]
 
        # SET UP MCMC
        max_ebv=1.0
        ndim, nwalkers, nchains, nburn = 3, 100, 200, 100
#        global count
#        count=0

        wha= (dd['id']=='halpha')
        whb= (dd['id']=='hbeta') 
        
        if (np.isnan(dd['flux'][whb]) | (dd['flux'][whb] ==-666)) :
            ebv_0 =  0.5*max_ebv
        else:
            ebv_0 = ext.calc_ebv(dd['flux'][wha]/dd['flux'][whb]) #even though already normalized
            
        if ebv_0<=0:
            ebv_0=1.e-3 #cannot be less than 0
        
        par=[0.5*(zrange[1]-zrange[0])+zrange[0], 0.5*(qrange[1]-qrange[0])+qrange[0], ebv_0]
        
        pos_start=self.generate_start(par, ndim, nwalkers, zrange, qrange, max_ebv)
#        pdb.set_trace()
        ##########################################################################

        if quiet ==False:
            print('starting the MCMC run with %d walkers, each chain %d samples long' \
                  % (nwalkers, nchains))
            
        random.seed(18)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, \
            args=(dd, good, flag, self.intergrid, zrange, qrange, max_ebv, idnorm))
       
        sampler.run_mcmc(pos_start, nchains)
    
        # CHAIN is an array of dimensions: nwalkers, nsteps, ndim
        samples = sampler.chain[:,nburn:, :].reshape((-1, ndim))
        
        ## flattening the chain... 
        samples[:,0]=samples[:,0]+self.logohsun
        # this function produces: median, upper error bar (84-50th percent), lower error bar (50-16th percent)                                                
        zz_mcmc, qq_mcmc, ebv_mcmc= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                     zip(*np.percentile(samples, [16, 50, 84],axis=0)))
 
#        chain = sampler.chain
#        pdb.set_trace()
#        tau = np.mean([integrated_time(walker) for walker in chain], axis=0)      
             
        if self.plot==True:
#             pdb.set_trace()
             fig = corner.corner(samples, \
                labels=[r"$\rm 12+log(O/H)$", r"$\rm log(q) \ [cm s^{-1}]$", r"$ \rm E(B-V)  \ [mag]$"],\
                show_titles=True,
                title_kwargs={"fontsize": 10},\
                label_kwargs={"fontsize": 14},\
                data_kwargs={"ms": 0.6})             
             self.fig=fig
             plt.show()
        
        if self.plot_z_steps==True:
             fig=plt.subplots()
             for i in range(nwalkers):
                 plt.plot(range(len(sampler.chain[i,:,0])),sampler.chain[i,:,0]+self.logohsun, c="gray", alpha=0.3)
                 plt.plot(range(nburn,len(sampler.chain[i,:,0])),sampler.chain[i,nburn:,0]+self.logohsun, c="red", alpha=0.3)
             plt.axhline(np.median(sampler.chain[:,nburn:,0])+self.logohsun, c="blue", ls="dashed")
             plt.axvline(nburn, c="blue", ls="dotted", alpha=0.3)
             plt.xlabel("Nstep")
             plt.ylabel("12+log(O/H)")
             plt.show()
        
        if self.plot_q_steps==True:
             fig=plt.subplots()
             for i in range(nwalkers):
                 plt.plot(range(len(sampler.chain[i,:,1])),sampler.chain[i,:,1], c="gray", alpha=0.3)
                 plt.plot(range(nburn,len(sampler.chain[i,:,1])),sampler.chain[i,nburn:,1], c="red", alpha=0.3)
             plt.axhline(np.median(sampler.chain[:,nburn:,1]), c="blue", ls="dashed")
             plt.axvline(nburn, c="blue", ls="dotted", alpha=0.3)
             plt.xlabel("Nstep")
             plt.ylabel("log(q)")
             plt.show()
         
        if self.plot_ebv_steps==True:
             fig=plt.subplots()
             for i in range(nwalkers):
                 plt.plot(range(len(sampler.chain[i,:,2])),sampler.chain[i,:,2], c="gray", alpha=0.3)
                 plt.plot(range(nburn,len(sampler.chain[i,:,2])),sampler.chain[i,nburn:,2], c="red", alpha=0.3)
             plt.axhline(np.median(sampler.chain[:,nburn:,2]), c="blue", ls="dashed")
             plt.axvline(nburn, c="blue", ls="dotted", alpha=0.3)
             plt.xlabel("Nstep")
             plt.ylabel("E(B-V)")
             plt.show()
             
        dd['zarr']=np.linspace(zrange[0]+self.logohsun, zrange[1]+self.logohsun, 20)
        dd['qarr']=np.linspace(qrange[0], qrange[1], 20)
        dd['ebvarr']=np.linspace(0, max_ebv, 20)
        
        dd['z_pdf'], dd['zarr'] = np.histogram(samples[:,0], density=True,bins= dd['zarr'])
        dd['q_pdf'], dd['qarr'] = np.histogram(samples[:,1], density=True,bins= dd['qarr'])
        dd['ebv_pdf'], dd['ebvarr'] = np.histogram(samples[:,2], density=True,bins= dd['ebvarr'])
        
        dd['zarr'] = (dd['zarr'][1:]+dd['zarr'][:-1])/2.
        dd['qarr'] = (dd['qarr'][1:]+dd['qarr'][:-1])/2.
        dd['ebvarr'] = (dd['ebvarr'][1:]+dd['ebvarr'][:-1])/2.
        
        dd['Z']=zz_mcmc[0]
        dd['err_up_Z']=zz_mcmc[1]
        dd['err_down_Z']=zz_mcmc[2]
        
        dd['q']=qq_mcmc[0]
        dd['err_up_q']=qq_mcmc[1]
        dd['err_down_q']=qq_mcmc[2]
        
        dd['ebv']=ebv_mcmc[0]
        dd['err_up_ebv']=ebv_mcmc[1]
        dd['err_down_ebv']=ebv_mcmc[2]
        
        dd['acc_fraction'] = np.mean(sampler.acceptance_fraction)

        #to find peaks in PDFs
        z_peaks, _ = find_peaks(dd['z_pdf'], prominence=(0.1, None))   
        q_peaks, _  = find_peaks(dd['q_pdf'], prominence=(0.1, None))   
        ebv_peaks, _  = find_peaks(dd['ebv_pdf'], prominence=(0.1, None))   
        
        # dx = (dd['zarr'][1] - dd['zarr'][0])/2.
        # dz_pdf = np.gradient(dd['z_pdf'],dx)
        # ddz_pdf = np.gradient( dz_pdf, dx )

        # dx = (dd['qarr'][1] - dd['qarr'][0])/2.
        # dq_pdf = np.gradient(dd['q_pdf'], dx)
        # ddq_pdf = np.gradient( dq_pdf, dx )

        # dx = (dd['ebvarr'][1] - dd['ebvarr'][0])/2.
        # debv_pdf = np.gradient(dd['ebv_pdf'], dx)
        # ddebv_pdf = np.gradient( debv_pdf, dx )
        
        # if self.plot_derivatives==True:
        #     fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(20,10))
        #     ax1.axhline(y=0,linestyle='dashed',color='magenta')
        #     ax1.scatter(dd['zarr'], dd['z_pdf'],color='k',s=25)
        #     ax1.plot(dd['zarr'], dz_pdf,color='green',label='Z_pdf I derivative')
        #     ax1.plot(dd['zarr'], ddz_pdf,color='red',label='Z_pdf II derivative')
        #     ax1.scatter(dd['zarr'][z_peaks], dd['z_pdf'][z_peaks],color='cyan',s=15,label='Z_pdf peaks')

        #     ax1.set_xlabel('12+log(O/H)')
        #     ax1.legend()
        #     ax2.axhline(y=0,linestyle='dashed',color='magenta')
        #     ax2.scatter(dd['qarr'], dd['q_pdf'],color='k')
        #     ax2.plot(dd['qarr'], dq_pdf,color='green',label='q_pdf I derivative')
        #     ax2.plot(dd['qarr'], ddq_pdf,color='red',label='q_pdf II derivative')
        #     ax2.scatter(dd['qarr'][q_peaks], dd['q_pdf'][q_peaks],color='cyan',s=15,label='q_pdf peaks')

        #     ax2.set_xlabel('log(q)')
        #     ax3.axhline(y=0,linestyle='dashed',color='magenta')
        #     ax3.scatter(dd['ebvarr'], dd['ebv_pdf'],color='k')
        #     ax3.plot(dd['ebvarr'], debv_pdf,color='green',label='ebv_pdf I derivative')
        #     ax3.plot(dd['ebvarr'], ddebv_pdf,color='red',label='ebv_pdf II derivative')
        #     ax3.scatter(dd['ebvarr'][ebv_peaks], dd['ebv_pdf'][ebv_peaks],color='cyan',s=15,label='ebv_pdf peaks')
        #     ax3.set_xlabel('E(B-V)')
        #     ax3.legend()
        #     plt.show()

        dd['flag'][0] = len(z_peaks)
        dd['flag'][1] = len(q_peaks)
        dd['flag'][2] = len(ebv_peaks)
        
        dd['Z_max'][0][0:len(z_peaks)] = dd['zarr'][z_peaks]
        dd['Z_max'][1][0:len(z_peaks)] = dd['z_pdf'][z_peaks]
        dd['q_max'][0][0:len(q_peaks)] = dd['qarr'][q_peaks]
        dd['q_max'][1][0:len(q_peaks)] = dd['q_pdf'][q_peaks]
        dd['ebv_max'][0][0:len(ebv_peaks)] = dd['ebvarr'][ebv_peaks]
        dd['ebv_max'][1][0:len(ebv_peaks)] = dd['ebv_pdf'][ebv_peaks]
       
        if (np.max(dd['z_pdf'][0:1]) >= 0.5*np.max(dd['z_pdf'])):
            dd['flag'][3] = 1
        if (np.max(dd['z_pdf'][-2:-1]) >= 0.5*np.max(dd['z_pdf'])):
            dd['flag'][3] = 2
        if (np.max(dd['z_pdf'][0:1]) >= 0.5*np.max(dd['z_pdf'])) & (np.max(dd['z_pdf'][-2:-1]) > 0.5*np.max(dd['z_pdf'])):
            dd['flag'][3] = 3
            
        if (np.max(dd['q_pdf'][0:1]) >= 0.5*np.max(dd['q_pdf'])):
            dd['flag'][4] = 1
        if (np.max(dd['q_pdf'][-2:-1]) >= 0.5*np.max(dd['q_pdf'])):
            dd['flag'][4] = 2
        if (np.max(dd['q_pdf'][0:1]) >= 0.5*np.max(dd['q_pdf'])) & (np.max(dd['q_pdf'][-2:-1]) > 0.5*np.max(dd['q_pdf'])):
            dd['flag'][4] = 3 

        if (np.max(dd['ebv_pdf'][0:1]) >= 0.5*np.max(dd['ebv_pdf'])):
            dd['flag'][5] = 1
        if (np.max(dd['ebv_pdf'][-2:-1]) >= 0.5*np.max(dd['ebv_pdf'])):
            dd['flag'][5] = 2
        if (np.max(dd['ebv_pdf'][0:1]) >= 0.5*np.max(dd['ebv_pdf'])) & (np.max(dd['ebv_pdf'][-2:-1]) > 0.5*np.max(dd['ebv_pdf'])):
            dd['flag'][5] = 3 
        
        # calculate chi-square
        w = np.isfinite(dd['flux']) & (dd['flux'] != -666)
        
        #read observed values
        fobs=dd['flux']
        eobs=dd['error']
        
        # fmod not corrected for reddening #not interpolating
        fmod = self.flux_grid_ext(self.intergrid, zz_mcmc[0]-self.logohsun, qq_mcmc[0], ebv_mcmc[0], idnorm)

        emod=dd['epsilon']*fmod

        chi2=np.nansum((fobs[w]-fmod[w])**2/(eobs[w]**2+emod[w]**2))/(len(fobs[w])-ndim) #reduced chi-square

        dd['chi2']=chi2

        line_info={'id':self.id, 'fobs':np.array(self.flux)+np.nan, 'fmod':np.array(self.flux)+np.nan, \
                   'eobs':np.array(self.flux)+np.nan, 'emod':np.array(self.flux)+np.nan, \
            'chi2_line':np.array(self.flux)+np.nan}
        
        for i in range(nlines_in):
              auxind=(dd['id']==self.id[i])
              line_info['fmod'][i]=self.flux_grid_ext(self.intergrid, zz_mcmc[0]-self.logohsun, qq_mcmc[0], ebv_mcmc[0], idnorm)[auxind]#grid.value_grid_at(grid0,zz_mcmc[0]-self.logohsun, qq_mcmc[0])[auxind]
              line_info['emod'][i]=dd['epsilon'][auxind]*line_info['fmod'][i]
              line_info['fobs'][i]=dd['flux'][auxind]
              line_info['eobs'][i]=dd['error'][auxind]
              if np.isfinite(dd['flux'][auxind]) & (dd['flux'][auxind] != -666):
                  line_info['chi2_line'][i]=(line_info['fobs'][i]-line_info['fmod'][i])**2/ \
                          (dd['error'][auxind]**2+(line_info['emod'][i])**2)
              else:
                  line_info['chi2_line'][i] = -666
        
        #pdb.set_trace()
        #WRITE THE SOLUTION as attribute to the IZI_MCMC class 
        
        self.sol=[]
        self.sol=dd
        self.samples=samples
        self.line_stat=[]
        self.line_stat=line_info
          
        #pdb.set_trace()
        
        if self.quiet ==False:
            print('12+log(O/H) (%f, + %f, - %f), q (%f, + %f, - %f), E(B-V) (%f, + %f, - %f)'\
              % (dd['Z'] ,dd['err_up_Z'] ,dd['err_down_Z'],\
              dd['q'], dd['err_up_q'], dd['err_down_q'], \
              dd['ebv'], dd['err_up_ebv'], dd['err_down_ebv'] ))
              
            print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))       
            
    @staticmethod
    def flux_grid_ext(grid_, zz, qq, ebv, idnorm):

        flux_at = grid.value_grid_at_no_interp(grid_, zz, qq)
        
        red_vec=ext.reddening_vector_calzetti(grid_['WAV'][0], ebv)
        flux_redd=flux_at/red_vec
        
        #NORMALISE GRID           
        norm=flux_redd[grid_['ID'][0] == idnorm ]
        
        flux_redd=flux_redd/norm


        return flux_redd
    
    #DEFINE STATISTICAL FUNCTIONS
    #LIKELIHOOD of one specific point of paramter space (theta)

    def lnlike(self, theta, dd, good, flag, grid_, idnorm):
        zz, qq, ebv = theta
        fff = np.array(self.flux_grid_ext(grid_,zz, qq, ebv, idnorm))
        ngood=good.sum()
        like=1.

        if self.logqprior is not None: 
            gauss_q = 1./(self.logqprior[1]*np.sqrt(2.*np.pi))*np.exp(- ( ( qq - self.logqprior[0] )/self.logqprior[1] )**2/2. ) 
        else:
            gauss_q=1.
 
        if self.logzprior is not None: 
            gauss_z = 1./(self.logzprior[1]*np.sqrt(2.*np.pi))*np.exp(- ( ( zz - self.logzprior[0] )/self.logzprior[1] )**2/2. ) 
        else:
            gauss_z=1.
               
        for j in range(ngood):
            if (flag[j] == 1):
                e2=dd['error'][good][j]**2.0 + (dd['epsilon'][good][j]*fff[good][j] )**2.0
                fdf2= (dd['flux'][good][j]- fff[good][j])**2.0
                like=like*1/np.sqrt(2*np.pi)*np.exp(-0.5*fdf2/e2)/np.sqrt(e2)*gauss_q*gauss_z
                
            if (flag[j] == 2):
                edf= (dd['error'][good][j]- fff[good][j])
                e2=dd['error'][good][j]**2.0 + (dd['epsilon'][good][j]*fff[good][j] )**2.0
                like=like*0.5*(1+special.erf(edf/np.sqrt(e2*2)))*gauss_q*gauss_z

        return np.log(like)
    
    #PRIOR
    def lnprior(self, theta, zrange, qrange, max_ebv):
        zz, qq, ebv = theta
        
        if zrange[0] < zz < zrange[1] and qrange[0] < qq < qrange[1] and \
           0. < ebv < max_ebv :
           return 0.0
        else:
            return -np.inf
        
    #POSTERIOR
    def lnprob(self, theta, dd, good, flag, grid_, zrange, qrange, max_ebv, idnorm):
        lp = self.lnprior(theta, zrange, qrange, max_ebv)
        
        tot=lp + self.lnlike(theta, dd, good, flag, grid_, idnorm)

        if not np.isfinite(tot):
            return -np.inf
        
        return tot
    
    def generate_start(self, par, ndim, nwalkers, zrange, qrange, max_ebv):
        pos_start=np.zeros((nwalkers, ndim))
        for i in range(nwalkers):
            
            trial=par + 0.5*np.random.randn(ndim) 
            if  trial[0] < zrange[0] or trial[0] > zrange[1] :
                 trial[0] = par[0]
#        
            if  trial[1] < qrange[0] or trial[1] > qrange[1] :
                 trial[1] = par[1]
#        
            if  trial[2] < 0. or trial[2] > max_ebv :
                 trial[2] = par[2]
         
            pos_start[i,:]=trial
    
        return pos_start


