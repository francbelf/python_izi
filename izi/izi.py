#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:13:01 2017
+
; NAME:
;
;   IZI 
;
; PURPOSE:
;
;   Compute the posterior PDF of the gas-phase metallicity and the
;   ionization parameter given a set of observed emission line fluxes
;   and errors, and a photo-ionization model grid. The code
;   interpolates the models to a finer grid and evaluates the
;   posterior PDF of the parameters given the data and the models.
;   
; CALLING SEQUENCE:
;
;  result = IZI(flux, error, id, gridfile=None, templ_dir='', grid=None,
                 plot=True, logohsun=None, epsilon=None, nz=None, nq=None, 
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
;    input emission line. Fluxes must be corrected for dust extinction.
;
;    ERROR: rray of emission line flux errors. Upper limits can be
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
;    PLOT: set this keyword to True to produce plots of the
;    parameters PDFs and of diagnostic line ratios versus log(O/H) and q.
;   
;    LOGOHSUN: set this keyword to a user suplied value which is used
;    instead of the LOGOHSUN value in the model grid file.
;
;    EPSILON: systematic uncertainty in dex for the emission line
;    fluxes in the model (see equation 3 in Blanc et al. 2014). If not
;    provided the code assumes a default value is 0.15 dex.
;
;    NZ: number of log(Z) elements in the interpolated grid, if not
;    provided the default is NZ=50
;   
;    NQ: number of log(q) elements in the interpolated grid, if not
;    provided the default is NZ=50
;
;    INTERGRIDFILE: string containing the filename for an already
;    interpolated photo-ionization model grid. If provided this file
;    is used intead of GRIDFILE and the interpolation step is
;    skiped. This interpolated grid file should be created using the
;    OUTGRIDFILE keyword. Useful for speeding computation time when
;    running IZI for a large sample of objects.
;
;    OUTGRIDFILE: string containing a filename to save the
;    interpolated grid file for latter use with INTERGRIDFILE
;
;    LOGZLIMITS: 2 element array conatining the lower and upper
;    log(Z) limits for the interpolated grid (section 3 of Blanc et
;    al. 2014) in units of 12+log(O/H)
;
;    LOGQLIMITS: 2 element array conatining the lower and upper
;    log(Z) limits for the interpolated grid (section 3 of Blanc et
;    al. 2014) in units of log(cm/s)
;
;    LOGZPRIOR: (Np x 2) element array especifying a prior for log(Z)
;    in 12+log(O/H) units. The first array [Np,0] contains Np
;    values for the metallicity and the second array [Np,1] contains
;    the probability for each value. This array is interpolated to the
;    NZ grid.
;
;    LOGQPRIOR: (Np x 2) element array especifying a prior for log(q)
;    in log(cm/s) units. The first array [Np,0] contains Np
;    values for the ionization parameter and the second array [Np,1] contains
;    the probability for each value. This array is interpolated to the
;    NQ grid.
;
;    NONORM: set this keyword to avoid the normalization of the line
;    fluxes. This is useful when using model grids of line ratios
;    instead of line fluxes.
;
;    QUIET: set to true to avoid printing the inferred Z and q to standard 
;    output
; 
; OUTPUT:
;
;    RESULT.sol: a dictionary containing the output best fit parameters and PDFs
;
;      id: names of the emission lines in the photo-ionization model
;      flux: nomalized input line fluxes, same size as id
;      error: error in normalized line fluxes, same size as id
;      post: 2D (nq*nz) PDF
;      chi2: chi^2 between observed fluxes and model fluxes at mode of the joint PDF
;      Z_joint: joint mode best-fit metallicity in units of 12+log(O/H)
;      err_down_Z_joint: upper 1 sigma error of 12+log(O/H) for joint PDF
;      err_up_Z_joint: lower 1 sigma error of 12+log(O/H) for joint PDF
;      q_joint: joint mode best-fit ionization parameter in units of log(cm/s)
;      err_down_q_joint: upper 1 sigma error on the above
;      err_up_q_joint: lower 1 sigma error on the above
;      Z_max, err_down_Z_max, err_up_Z_max: marginalized best-fit metallicity 
;          in units of 12+log(O/H) and errors
;      Z_mean,err_down_Z_mean, err_up_Z_mean: marginalized mean of the metallicity 
;          PDF in units of 12+log(O/H) and errors
;      zarr: array of interpolated metallicity values in units of 12+log(O/H)
;      z_pdf: 1D marginalized metallicity PDF as a function of ZARR
;      q_max, err_down_q_max, err_up_q_max: marginalized best-fit ionization parameter 
;          in units of log(cm/s) and errors
;      q_mean,err_down_q_mean, err_up_q_mean: marginalized mean of the ionization parameter 
;          PDF in units of log(cm/s) and errors
;      qarr=array of interpolated ionization parameter values in units of log(cm/s)
;      q_pdf=marginalized metallicity PDF as a function of qarr
;
;    RESULT.line_stat: a dictionary containing the best model fit for each line and 
;      other diagnostic information 
;          
;      id: names of the emission lines, same size and *same order* as the input flux array
;      fobs: normalised emission line flux, same size and *same order* as the input flux array
;      fmod: normalised best fit model flux, same size and *same order* as the input flux array
;      chi2_line: chi2 for each emission line, same size and *same order* as the input flux array
;
; USAGE EXAMPLE:
;    
;   templ_dir='/Volumes/fbdata/CODE/python_MaNGA_tools/MPL6/analyse/izi'
;
;  #THIS TEST DATA IS TAKEN FROM THE Berg 2015 -35.9, 57.7
;   flux=[3.44,2.286, 1, 0.553, 0.698, 2.83]
;    error=[0.17, 0.07,0.05, 0.23, 0.027, 0.15]
;   id=['oii3726;oii3729','oiii4959;oiii5007', 'hbeta' , 'nii6548;nii6584', \
;    'sii6717;sii6731', 'halpha']
;
;   out=izi(flux, error, id, \
;        templ_dir=templ_dir,\
;        intergridfile=templ_dir+'/grids/interpolgrid_50_50l09_high_csf_n1e2_6.0Myr.fits', \
;        epsilon=0.2, quiet=False, plot=True)   
;
;  joint Z, q 8.33982294555 7.49042662796
;  mode Z (8.372518, + 0.130780, - 0.098085), q (7.490427, + 0.098085, - 0.098085)
;  mean Z, (8.380919, + 0.122379, - 0.106487), q (7.517959, + 0.070553, - 0.125618)
;     
;
; MODIFICATION HISTORY: 
;
;    V1.0 - Created by G.A. Blanc
;    V2.0 - Translated into Python by F. Belfiore, 23 Jan 2018
;    V2.1 - (25/03/18 FB) Final tweaks to the Python version and updated the docstring. 
;    V3.0 - Upgraded to Python3 by M. Mingozzi
; =====================================================================================

@author: francesco
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.table as table
from scipy import interpolate
from scipy.integrate import simps
from scipy import special
import pdb
from os import path
import time



#%%
def read_grid(gridfile):
    grid=table.Table.read(gridfile)
    return grid

def uprior(xrang):
# calculates a uniform prior for x
# recieves as input the range of allowed x values (2 element array)
  return 1.0/(float(xrang[1])-xrang[0])

def userprior(x, xarr, yarr):
#FUNCTION userprior, x, xarr, yarr
#; interpolates a user provided prior (xarr, yarr) to x
#; returns 0 if x is outside range of xarr
    if (np.max(x) <= np.min(xarr)) or (np.min(x) >= np.max(xarr)):
        return 0
    else:
        f=interpolate.interp1d(xarr, yarr)
        return f(x)
        
def make_grid_2d(grid0, flux):

    qauxarr=np.unique(grid0['LOGQ'])
    zauxarr=np.unique(grid0['LOGZ'])
    nlines0=len(grid0['ID'][0])
    fflux=np.zeros( (len(qauxarr),len(zauxarr), nlines0))
    for j in range(nlines0):     
        for iq in range(len(qauxarr)):
            for iz in range(len(zauxarr)):
                w = (grid0['LOGZ']==zauxarr[iz]) & (grid0['LOGQ']==qauxarr[iq])
                fflux[iq,iz,j]=  flux[w,j]      
    return fflux

def make_img_2d(grid0, flux):

    qauxarr=np.unique(grid0['LOGQ'])
    zauxarr=np.unique(grid0['LOGZ'])
    fflux=np.zeros( (len(qauxarr),len(zauxarr)))
    for iq in range(len(qauxarr)):
        for iz in range(len(zauxarr)):
            w = (grid0['LOGZ']==zauxarr[iz]) & (grid0['LOGQ']==qauxarr[iq])
            fflux[iq,iz]=  flux[w]      
    return fflux   




#%%
    
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
#        qauxarr=np.unique(self.grid0['LOGQ'])
#        zauxarr=np.unique(self.grid0['LOGZ'])
        logzin=np.array(self.grid0['LOGZ'])
        logqin=np.array(self.grid0['LOGQ'])

        for i in range(nlines0): 
             fluxarr[:,:,i]=interpolate.griddata( (logzin,logqin),
                    flux[:,i], (grid_x, grid_y), method='cubic')
        #   ALTERNATIVE INTERPOLATION SCHEME                 
        #                 f= interpolate.interp2d(zauxarr,qauxarr, fflux[:,:,i], kind='cubic')
        #                 fluxarr2[:,:,i]=f(zarr, qarr)
           
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


class izi(object):
    
    def __init__(self, flux, error, id, gridfile=None, templ_dir=None,
                 logzlimits=None, logqlimits=None, 
                 epsilon=0.15, nz=50, nq=50,
                 intergridfile=None, integrid=None, outgridfile=False, 
                 logzprior=None, logqprior=None,  nonorm=0,
                 quiet=False, plot=True):

          
        #DECLARE INPUT TO SELF
        self.flux = flux          #  flux array
        self.error = error       # error flux array
        self.id = id  # IDs of different emission lines
        self.gridfile = gridfile
        self.templ_dir = templ_dir
        self.logzlimits = logzlimits
        self.logqlimits = logqlimits
        self.intergridfile=intergridfile
        self.intergrid=integrid
        self.nonorm=nonorm
        self.outgridfile=outgridfile
        self.logzprior=logzprior
        self.logqprior=logqprior
        self.plot=plot
        self.nz=nz
        self.nq=nq
        self.quiet=quiet
        self.epsilon=epsilon
        
        nlines_in=len(self.flux)
        assert len(self.error) == nlines_in and len(self.id) == nlines_in, \
        'ERROR Flux, Error, and ID arrays do not have the same number of elements'

        # INPUT FILES CHECKING    
        # IF NOT SPECIFIED BY USER USE DEFAULT Levesque models with density 
        # 10^2 cm-3, composite SF, and 6Myr age          
        if self.gridfile == None:
            self.gridfile='l09_high_csf_n1e2_6.0Myr'
            #  self.gridfile='d13_kappa20'
        else:
            self.gridfile=gridfile
        if self.templ_dir==None:
            self.templ_dir = path.dirname(path.realpath(__file__))[:-4]+'/grids/'
        
        
        if self.intergridfile == None:
            # PREPARE ORIGINAL GRID
            
            # READ GRID using the grid class
            grid0=grid(self.gridfile, templ_dir=self.templ_dir)
            
            # APPLY LIMITS to grid        
            grid0.apply_limits(logzlimits=self.logzlimits, logqlimits=self.logqlimits)
            #     
            self.logohsun=grid0.logohsun
            nlines0=len(grid0.grid0['ID'][0])
            #  number of steps in log(Z) * number of steps in log(q)
#            ngrid0=len(grid0['LOGZ'])
                
            
            #INTERPOLATE GRID
#            pdb.set_trace()
            grid0.interpolate_grid(nz=self.nz, nq=self.nq)
            
            self.intergrid=grid0.intergrid
            
            #DEFINE PARAMTERS OF GRID
            zarr=np.linspace(np.min(self.intergrid['LOGZ']), np.max(self.intergrid['LOGZ']), self.nz)
            qarr=np.linspace(np.min(self.intergrid['LOGQ']), np.max(self.intergrid['LOGQ']), self.nq)
            nintergrid=len(self.intergrid['ID'])
            # WRITE INTERPOLATED GRID IF USER WANTS TO
            if self.outgridfile ==True:
                a=self.intergrid
                a.write(self.templ_dir+'/interpolgrid_'+str(self.nz)+'_'+\
                        str(self.nq)+self.gridfile+'.fits', overwrite=True) 
        else:
            # READ GRID using the grid class
            grid0=grid(self.intergridfile, templ_dir=self.templ_dir)
            
            self.intergrid=grid0.grid0
            nintergrid=len(self.intergrid['ID'])
            nlines0=len(self.intergrid['ID'][0])
            self.nz=len(np.unique(self.intergrid['LOGZ']))
            self.nq=len(np.unique(self.intergrid['LOGQ']))
            
            zarr=np.linspace(np.min(self.intergrid['LOGZ']), np.max(self.intergrid['LOGZ']), self.nz)
            qarr=np.linspace(np.min(self.intergrid['LOGQ']), np.max(self.intergrid['LOGQ']), self.nq)
            
            self.logohsun=grid0.logohsun


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

        # INCLUDE SYSTEMATIC UNCERTAINTY IN THE PHOTO-IONIZATION MODELS
        # default is 0.15 dex systematic uncertainty                   
        epsilon2=epsilon*np.log(10)           
      
#; CREATE DATA STRUCTURE CONTAINING LINE FLUXES AND ESTIMATED PARAMETERS
# note that callinf this d messes us the python debugger:(
        dd={'id':self.intergrid['ID'][0],       # line id
           'flux':np.zeros(nlines0)-999,      # line flux      
           'error':np.zeros(nlines0)-999}
                 
#FILL STRUCTURE WITH LINE FLUXES
#        
        for i in range(nlines_in):
              auxind=(dd['id'] == self.id[i])
              nmatch=auxind.sum()

              assert nmatch == 1, 'ERROR: ===== Line ID '+self.id[i]+'not recognized ====='
              dd['flux'][auxind]=self.flux[i]
              dd['error'][auxind]=self.error[i]              
        
        
#  INDEX OF LINES WITH MEASUREMENTS
        good=(dd['error'] != -999)
        ngood=good.sum()
        measured=(dd['flux'] != -999)
#        nmeasured=measured.sum()
        upperlim=(dd['error'] != -999) & (dd['flux'] == -666)
#        nupper=upperlim.sum()
        
        flag0=np.zeros(nlines0)
        flag0[measured]=1      #measured flux
        flag0[upperlim]=2      #upper limit on flux
#        this array has length ngood, which is the number of lines with 
#        given error measurements. If error is given but no flux this is treated
#        as an upper limit
        flag=flag0[good]
        
# ; NORMALIZE LINE FLUXES TO H-BETA OR
# ; IF ABSENT NORMALIZE TO BRIGHTEST LINE
#        pdb.set_trace()
        if self.nonorm ==0: #; use nonorm for line ratio fitting
            idnorm='hbeta'
            in_idnorm= (dd['id']==idnorm)
            
            if (dd['flux'][in_idnorm] ==-999):
                a=dd['id'][measured]
                idnorm=a[[np.argmax(dd['flux'][measured])]]
                in_idnorm= (dd['id']==idnorm)
            
            norm=dd['flux'][in_idnorm]
#            print 'flux of', idnorm, 'is', norm
#            NORMALISE INPUT FLUXES
            dd['flux'][measured]=dd['flux'][measured]/norm[0]
            dd['error'][good]=dd['error'][good]/norm[0]
#            NORMALISE GRID           
            norm=self.intergrid['FLUX'][:,self.intergrid['ID'][0] == idnorm ]
            
            self.intergrid['FLUX']=self.intergrid['FLUX']/norm
        fff=np.array(self.intergrid['FLUX'])
     
# CALCULATE LIKELIHOOD AND POSTERIOR
# if using EMCEE THIS PART NEEDS TO BE REPLACED BY THE SAMPLER

        like=np.zeros(nintergrid)+1.0
        post=np.zeros(nintergrid)+1.0
        zrange=[np.min(self.intergrid['LOGZ']), np.max(self.intergrid['LOGZ'])]
        qrange=[np.min(self.intergrid['LOGQ']), np.max(self.intergrid['LOGQ'])]
#  
        for i in range(nintergrid):
            for j in range(ngood):
                if (flag[j] == 1):
                    e2=dd['error'][good][j]**2.0 + (epsilon2*fff[i, good][j] )**2.0
                    fdf2= (dd['flux'][good][j]- fff[i, good][j])**2.0
                    like[i]=like[i]/np.sqrt(2*np.pi)*np.exp(-0.5*fdf2/e2)/np.sqrt(e2)
                    
                if (flag[j] == 2):
                    edf= (dd['error'][good][j]- fff[i, good][j])
                    e2=dd['error'][good][j]**2.0 + (epsilon2*fff[i, good][j] )**2.0
                    like[i]=like[i]*0.5*(1+special.erf(edf/np.sqrt(e2*2)))
#                    print 'upper limit'
   
#CALCULATE POSTERIOR BY INCLUDING PRIORS AND NORMALIZING
#                    USE of custom priors has not been tested
# CHANGE LOGZPRIOR TO SOLAR UNITS
            if self.logzprior !=None:
                self.logzprior[:,0]=self.logzprior-self.logohsun  
            if (self.logzprior == None) and (self.logqprior == None):
                post[i]=uprior(zrange)*uprior(qrange)*like[i]
            if (self.logzprior != None) and (self.logqprior == None):
                post[i]=userprior(self.intergrid['LOGZ'][i], self.logzprior[:,0], logzprior[:,1])*\
                 uprior(qrange)*like[i]
            if (self.logzprior == None) and (self.logqprior != None):
                post[i]=uprior(zrange)*\
                 userprior(self.intergrid['LOGQ'][i], self.logqprior[:,0], logqprior[:,1])*like[i]
            if (self.logzprior != None) and (self.logqprior != None):
                post[i]=userprior(self.intergrid['LOGZ'][i], self.logzprior[:,0], logzprior[:,1])*\
                userprior(self.intergrid['LOGQ'][i], self.logqprior[:,0], logqprior[:,1])*like[i]


# Nobody likes undefined and infinite likelihoods or posteriors
#
        like[np.isfinite(like)==0]==0
        post[np.isfinite(post)==0]==0
        like=np.array(like)
        post=np.array(post)
        
        #; WRITE JOINT PDFS        
        dd['zarr']=zarr
        dd['qarr']=qarr
        dd['post']=post
#        
        if np.sum(post)!=0:
            conf_int=[0.683, 0.955, 0.997]
    # SORT LIKELIHOOD AND POSTERIOR FOR GETTING BEST-FIT VALUES AND CONFIDENCE INTERVALS
    #        sort likelihood and posterior in descending order (highest likelihood first)
            sortlike= np.sort(like)[::-1]
            sortpost= np.sort(post)[::-1]
    #       generate arrays of sorted LogZ and LogQ according to the sorted posterior
            inds = post.argsort()
            sortz=self.intergrid['LOGZ'][inds][::-1]
            sortq=self.intergrid['LOGQ'][inds][::-1]
            sumlike=np.zeros(len(sortlike))
            sumpost=np.zeros(len(sortlike))
            for i in range(len(sortlike)):
                sumlike[i]=np.sum(sortlike[0:i+1])/np.sum(sortlike)
                sumpost[i]=np.sum(sortpost[0:i+1])/np.sum(sortpost) 
                
    #        CALCULATE BEST FIT METALLICITY, IONIZATION PARAMETER from JOINT POSTERIORS
    #        errors are given by the joint PDF confidence intervals    
    #        THESE error definitions implement the shortest interval method
    #        eg. sec 2.5.2 Andrae 2010
            short_int_jz=sortz[sumpost <= conf_int[0]]
            if len(short_int_jz) > 0:
                min_short_int_jz=np.min(short_int_jz)
                max_short_int_jz=np.max(short_int_jz)
            else:
                min_short_int_jz=0.
                max_short_int_jz=0.
                
            short_int_jq=sortq[sumpost <= conf_int[0]]
            if len(short_int_jq) > 0:
                min_short_int_jq=np.min(short_int_jq)
                max_short_int_jq=np.max(short_int_jq) 
            else:
                min_short_int_jq=0.
                max_short_int_jq=0.   
                
            dd['Z_joint']=sortz[0]+self.logohsun
            dd['err_down_Z_joint']=sortz[0]-min_short_int_jz
            dd['err_up_Z_joint']=max_short_int_jz-sortz[0]
            dd['q_joint']=sortq[0]
            dd['err_down_q_joint']=sortq[0]-min_short_int_jq
            dd['err_up_q_joint']=max_short_int_jq-sortq[0]
            
    #       COMPUTE chi2
            bestgrid= (self.intergrid['LOGZ'] == sortz[0]) & (self.intergrid['LOGQ'] == sortq[0])
            fobs=dd['flux'][ dd['flux'] > -666]
            eobs=dd['error'][ dd['flux'] > -666]
            fmod=np.squeeze(self.intergrid[bestgrid]['FLUX'])[dd['flux'] > -666]
            emod=epsilon2*fmod
            chi2=np.sum((fobs-fmod)**2/(eobs**2+emod**2))/len(fobs)
            dd['chi2']=chi2
            
            aa={'id':self.id, 'fobs':np.array(self.flux)+np.nan, 'fmod':np.array(self.flux)+np.nan, \
                'chi2_line':np.array(self.flux)+np.nan}
            for i in range(nlines_in):
                  auxind=( dd['id']==self.id[i])
                  aa['fmod'][i]=np.squeeze(self.intergrid[bestgrid]['FLUX'])[auxind]
                  aa['fobs'][i]=dd['flux'][auxind]
                  aa['chi2_line'][i]=(aa['fobs'][i]-aa['fmod'][i])**2/ \
                          (dd['error'][auxind]**2+(epsilon2*aa['fmod'][i])**2)
    
    # posterior for Z, marginalizing over q  
            postz=np.zeros(self.nz) 
            for j in range(self.nz):
                qq=self.intergrid['LOGQ']
                zz=self.intergrid['LOGZ']
#                pdb.set_trace()
    #            integrated over q at fixed z (zz==zarr[j])
                postz[j]=simps(post[zz==zarr[j]], qq[zz==zarr[j]] )
    #            normalize
            postz=postz/np.sum(postz)
            
            sort_post_z=np.sort(postz)[::-1]
            inds = postz.argsort()
            sortz_z=zarr[inds][::-1]
    #        WRITE MARGINALISED PDF for Z
            dd['z_pdf']=postz
            
    
            sumpost_sort_z=np.zeros(len(sort_post_z))
            # cumulative posterior, sorted
            for i in range(self.nz):
                sumpost_sort_z[i]=np.sum(sort_post_z[0:i+1])
                
            dd['Z_max']=zarr[postz == np.max(postz)]+self.logohsun # max of PDF    
            dd['Z_mean']=np.sum(zarr*postz)/np.sum(postz)+self.logohsun#  first momment of PDF
                  
    #        These errors are NOT the same as the ones quoted in Blanc 2014
    #        to revert to the definitions in Blanc 2014 use dd['Z_joint'] in the four 
    #        lines below instead of Z_mode and Z_mean respectively
    #        BTW, this implements the SYMMETRIC error defintion
    #        
    #        dd['err_down_Z_max']=dd['Z_max']-self.logohsun-zarr[sumpostz >= (1-0.683)/2][0]
    #        dd['err_up_Z_max']=zarr[sumpostz >= 1.0-(1-0.683)/2][0]-dd['Z_max']+self.logohsun
    #        dd['err_down_Z_mean']=dd['Z_mean']-self.logohsun-zarr[sumpostz >= (1-0.683)/2][0]
    #        dd['err_up_Z_mean']=zarr[sumpostz >= 1.0-(1-0.683)/2][0]-dd['Z_mean']+self.logohsun
                   
    #        THESE are the shortest interval error defintion
            short_int=sortz_z[sumpost_sort_z <= conf_int[0]]
            if len(short_int) > 0:
                min_short_int=np.min(short_int)
                max_short_int=np.max(short_int)
            else:
                min_short_int=0.
                max_short_int=0.
            
            dd['err_down_Z_max']=dd['Z_max']-self.logohsun-min_short_int
            dd['err_up_Z_max']=max_short_int-dd['Z_max']+self.logohsun
            dd['err_down_Z_mean']=dd['Z_mean']-self.logohsun-min_short_int
            dd['err_up_Z_mean']=max_short_int-dd['Z_mean']+self.logohsun
            
    #posterior for q marginalised over Z     
            postq=np.zeros(self.nq) 
            for j in range(self.nq):
                qq=self.intergrid['LOGQ']
                zz=self.intergrid['LOGZ']
                postq[j]=simps(post[qq==qarr[j]], zz[qq==qarr[j]] )
            postq=postq/np.sum(postq)
    #        WRITE MARGINALISED PDF for Q
            dd['q_pdf']=postq
            
            sort_post_q=np.sort(postq)[::-1]
            inds = postq.argsort()
            sortq_q=qarr[inds][::-1]
            
            sumpost_sort_q=np.zeros(len(sort_post_q))
            # cumulative posterior, sorted
            for i in range(self.nq):
                sumpost_sort_q[i]=np.sum(sort_post_q[0:i+1])
            
    
            dd['q_max']=qarr[postq == np.max(postq)] # max of PDF    
            dd['q_mean']=np.sum(qarr*postq)/np.sum(postq)#  first momment of PDF
            
    #        These errors are not actually the same as the ones quoted in Blanc 2014
    #        to revert to the definitions in Blanc 2014 use dd['Z_joint'] in the four 
    #        lines below instead of Z_mode and Z_mean respectively
    #        dd['err_down_q_max']=dd['q_max']-qarr[sumpostq >= (1-0.683)/2][0]
    #        dd['err_up_q_max']=qarr[sumpostq >= 1.0-(1-0.683)/2][0]-dd['q_max']
    #        dd['err_down_q_mean']=dd['q_mean']-qarr[sumpostq >= (1-0.683)/2][0]
    #        dd['err_up_q_mean']=qarr[sumpostq >= 1.0-(1-0.683)/2][0]-dd['q_mean']
            
            short_int=sortq_q[sumpost_sort_q <= conf_int[0]]
            if len(short_int) > 0:
                min_short_int=np.min(short_int)
                max_short_int=np.max(short_int)
            else:
                min_short_int=0.
                max_short_int=0.   
                
            dd['err_down_q_max']=dd['q_max']-min_short_int
            dd['err_up_q_max']=max_short_int-dd['q_max']
            dd['err_down_q_mean']=dd['q_mean']-min_short_int
            dd['err_up_q_mean']=max_short_int-dd['q_mean']
    
    #  PRINT OUTPUT  
            if self.quiet ==False:
                print('joint Z, q', dd['Z_joint'], dd['q_joint'])
                print('mode Z (%f, + %f, - %f), q (%f, + %f, - %f)' % (dd['Z_max'] ,dd['err_up_Z_max'] ,\
                              dd['err_down_Z_max'], dd['q_max'], dd['err_up_q_max'], dd['err_down_q_max']))
                print('mean Z, (%f, + %f, - %f), q (%f, + %f, - %f)' % (dd['Z_mean'] ,dd['err_up_Z_mean'] ,\
                              dd['err_down_Z_mean'], dd['q_mean'], dd['err_up_q_mean'], dd['err_down_q_mean']))
            
    # WRITE THE SOLUTION as attribute to the IZI class 
            self.sol=[]
            self.sol=dd
            self.line_stat=[]
            self.line_stat=aa
            
            
            if self.plot ==True:
                print('test')
                plt.figure(figsize=(10,4))
                ax1 = plt.subplot(131)
        
                #JUST A QUICK test: plot the likelihood function        
                aaa=post/np.sum(post)
      
                aout=make_img_2d(self.intergrid, aaa)
                ax1.imshow(aout, origin='lower',extent=[zrange[0]+self.logohsun,
                    zrange[1]+self.logohsun,qrange[0], qrange[1]],
                    aspect=(zrange[1]-zrange[0])/(qrange[1]-qrange[0]))
                ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
                ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
                ax1.set_title('P(Z,q)d(logZ)d(logq)')
                ax1.set_xlabel('12+log(O/H)')
                ax1.set_ylabel('log(q)') 
                circle1 = plt.Circle((dd['Z_joint'], dd['q_joint']), 0.02, color='r')
                ax1.add_artist(circle1)
                
                ax2 = plt.subplot(132)
                ax2.plot(zarr+self.logohsun, postz, c='k')
                ax2.set_ylabel('P(Z)d(logZ)')
                ax2.set_xlabel('12+log(O/H)')
                ax2.plot(np.linspace(0,50)*0.0+dd['Z_joint'],
                         np.linspace(0, np.max(postz)*1.1, 50), c='r', ls='-', label='Z joint')
                ax2.plot(np.linspace(0,50)*0.0+dd['err_up_Z_max']+dd['Z_max'],
                         np.linspace(0, np.max(postz)*1.1, 50), c='b', ls='--')
                ax2.plot(np.linspace(0,50)*0.0-dd['err_down_Z_max']+dd['Z_max'],
                         np.linspace(0, np.max(postz)*1.1, 50), c='b', ls='--')
                ax2.plot(np.linspace(0,50)*0.0+dd['Z_max'],
                         np.linspace(0, np.max(postz*1.1), 50), c='b', ls='-', label='Z max')
                ax2.plot(np.linspace(0,50)*0.0+dd['Z_mean'],
                         np.linspace(0, np.max(postz)*1.1, 50), c='g', ls='-', label='Z mean')
                ax2.legend()
                ax2.set_ylim(0, max(postz)*1,1)
                ax2.set_xlim(zarr.min()+self.logohsun, zarr.max()+self.logohsun)
                asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
                ax2.set_aspect(asp)
                
                ax3 = plt.subplot(133)
                ax3.plot(qarr, postq, c='k')
                ax3.set_ylabel('P(q)d(logq)')
                ax3.set_xlabel('log(q)')
                ax3.plot(np.linspace(0,50)*0.0+dd['q_joint'],
                         np.linspace(0, np.max(postq)*1.1, 50), c='r', ls='-', label='q joint')
                ax3.plot(np.linspace(0,50)*0.0+dd['q_max'],
                         np.linspace(0, np.max(postq)*1.1, 50), c='b', ls='-', label='q mode')
                ax3.plot(np.linspace(0,50)*0.0+dd['q_mean'],
                         np.linspace(0, np.max(postq)*1.1, 50), c='g', ls='-', label='q mean')
                ax3.plot(np.linspace(0,50)*0.0+dd['err_up_q_max']+dd['q_max'],
                         np.linspace(0, np.max(postq*1.1), 50), c='b', ls='--')
                ax3.plot(np.linspace(0,50)*0.0-dd['err_down_q_max']+dd['q_max'],
                         np.linspace(0, np.max(postq)*1.1, 50), c='b', ls='--')
                ax3.legend()
                ax3.set_ylim(0, max(postq)*1.1)
                ax3.set_xlim(qarr.min(), qarr.max())
                asp = np.diff(ax3.get_xlim())[0] / np.diff(ax3.get_ylim())[0]
                ax3.set_aspect(asp)
                
                plt.subplots_adjust(wspace=0.35)
        
        else:
            
#            IF the posterior is zero everywhere, then set all output to nan
            dd['Z_joint']=np.nan
            dd['err_down_Z_joint']=np.nan
            dd['err_up_Z_joint']=np.nan
            dd['q_joint']=np.nan
            dd['err_down_q_joint']=np.nan
            dd['err_up_q_joint']=np.nan
#            dd['post']=post*np.nan
            dd['z_pdf']=zarr*np.nan
            dd['q_pdf']=qarr*np.nan
            
            aa={'id':self.id, 'fobs':np.array(self.flux)+np.nan, 'fmod':np.array(self.flux)+np.nan, \
                'chi2_line':np.array(self.flux)+np.nan}
            dd['chi2']=np.nan
            
            dd['Z_max']= np.nan 
            dd['Z_mean']=np.nan
            dd['err_down_Z_max']=np.nan
            dd['err_up_Z_max']=np.nan
            dd['err_down_Z_mean']=np.nan
            dd['err_up_Z_mean']=np.nan
            
            dd['q_max']= np.nan 
            dd['q_mean']=np.nan
            dd['err_down_q_max']=np.nan
            dd['err_up_q_max']=np.nan
            dd['err_down_q_mean']=np.nan
            dd['err_up_q_mean']=np.nan
            self.sol=[]
            self.sol=dd
            self.line_stat=[]
            self.line_stat=aa
            raise ArithmeticError('IZI failed to fit data to model, try increasing the errors')

