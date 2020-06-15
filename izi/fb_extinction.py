# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
Provides a set of utility functions to deal with dust extinction.

*Revision history*:
    | **02 Jun 2016**: Original implementation by K. Westfall (KBW).
        Drawn from dust.py in David Wilkinson's FIREFLY code, and the
        dereddening functions in IDLUTILS.
    | **14 Jul 2016**: (KBW) Added :func:`apply_reddening`
    | **02 Dec 2016**: (KBW) Added :class:`GalacticExtinction`
    | **21 Aug 2018**: (FB) Reshaped to match current use

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np


def reddening_vector_calzetti(wave, ebv, rv=None):
    r"""
    Return the Calzetti et al. (2000) reddening vector.

    Args:
        wave (array-like): Wavelengths at which to calculate the
            reddening vector curve in angstroms.
        ebv (float): E(B-V) reddening used to normalize the curve.
        rv (float): (**Optional**) Ratio of V-band extinction to the B-V
            reddening:

            .. math:: 

                R_V = \frac{A_V}{E(B-V)}

            Default is 4.05.  Typical value for the diffuse ISM of the
            Milky Way is 3.1.

    Returns:
        One-dimensional array with the reddening vector that can be used 
        to deredden a spectrum by calculating:

        .. math::

            F(\lambda) = a(\lambda) f(\lambda)

        where :math:`a` is the vector returned by this function,
        :math:`f` is the observed flux, and :math:`F` is the dereddened
        flux.
    """
    # Check shapes
    ebv=float(ebv)
    _wave = np.atleast_1d(wave)
    if len(_wave.shape) != 1:
        raise ValueError('Must only provide a single wavlength vector.')
    if not isinstance(ebv, float):
        raise TypeError('Input reddening value must be a single float.')
    
    if rv is None:
        _rv = 4.1  
    else:
        _rv = rv

    lam = 1e4/_wave  # Convert Angstrom to micrometres and take 1/lambda
    rv = 4.05  # C+00 equation (5)

    # C+00 equation (3) but extrapolate for lam > 2.2
    # C+00 equation (4) but extrapolate for lam < 0.12
    k1 = np.where(lam >= 6300,
                  _rv + 2.659*(1.040*lam - 1.857),
                  _rv + 2.659*(1.509*lam - 0.198*lam**2 + 0.011*lam**3 - 2.156))
    fact = 10**(0.4*ebv*k1.clip(0))  # Calzetti+00 equation (2) 
    return fact  # The model spectrum has to be multiplied by this vector

def calc_ebv(obs_ratio, form='Calzetti', th_ratio = 2.86, obs_ratio_err=None):
    r"""
    Return the EVB value from the HA/Hb ratio given the intrinsic ratio and the extinction curve

    Args:
        obs_ratio (float):The observed ratio of Ha/Hb
        th_ratio (float): The intrinsic ration between Ha/Hb. Default: 2.86
        curve (string): (**Optional**) The extinction curve to use, default is Calzetti
        obs_ratio_err (float): (**Optional**): error in the observe line ratio

    Returns:
        numpy.ndarray : Returns the EBV value and the error if obs_ratio_err is provided
    """

    w_ha=6563.
    w_hb=4861.
    
    kha=2.5*np.log10(reddening_vector_calzetti(w_ha, 1.0))  
    khb=2.5*np.log10(reddening_vector_calzetti(w_hb, 1.0)) 
#    obtain kHa and kHb by computing the reddening vectors for EBV=1
    
    ebv= 2.5* np.log10(obs_ratio/th_ratio)/(khb-kha)
#    this is the correct EBV equation (see for example my PhD thesis Eq. 1.24)
    
    if obs_ratio_err==None:
        return ebv
    else:
#        this result is from simple error propagation of the EBV formula
#        remember EBV= 2.5/(Khb-Kha) * (log10(Obs ratio)- log10(theoretical ratio))
#        therefore err_EBV = 2.5/(Khb-Kha) * (err_Obs_ratio)/(Obs_ratio * ln10)
        e_ebv=2.5*obs_ratio_err/( (khb-kha) * np.log(10) *obs_ratio)
        return ebv, e_ebv


def calz_unred(wave, flux, ebv, e_flux=None, e_ebv=None, rv=None):
    r"""
    Return a dereddened flux using the Calzetti reddening vector.

    Args:
        wave (array-like): Wavelengths of the flux vector
        flux (array-like): flux vector to deredded    
        ebv (float): E(B-V) reddening used to normalize the curve.
        e_ebv (float): (**Optional**) error on E(B-V)
        rv (float): (**Optional**) Ratio of V-band extinction to the B-V
            reddening:

            .. math:: 

                R_V = \frac{A_V}{E(B-V)}

            Default is 4.05.

    Returns:
        numpy.ma.MaskedArray: dereddened flux and uncertainty (is either the 
        error on ebv or the error on the flux are provided)
    """      
        
#    get the dereddened flux  
    a=reddening_vector_calzetti(wave, ebv, rv=None)   
    flux_dered=a*flux
    
    if e_flux==None:
        return flux_dered
    elif e_flux!=None and e_ebv!= None:
        #    get the error on the dereddened flux     
        klam=2.5*np.log10(reddening_vector_calzetti(wave, 1.0))    

        s_f=0.4* klam*np.log(10)*e_ebv
        e_flux_dered = ( (e_flux/flux)**2 + s_f**2)**0.5 * flux_dered
        return flux_dered, e_flux_dered
    elif e_flux!=None and e_ebv==None:
        e_ebv=0.
        #    get the error on the dereddened flux     
        klam=2.5*np.log10(reddening_vector_calzetti(wave, 1.0))    

        s_f=0.4* klam*np.log(10)*e_ebv
        e_flux_dered = ( (e_flux/flux)**2 + s_f**2)**0.5 * flux_dered
        return flux_dered, e_flux_dered
    elif e_ebv!=None and e_flux==None:
        e_flux=0.
        #    get the error on the dereddened flux     
        klam=2.5*np.log10(reddening_vector_calzetti(wave, 1.0))    

        s_f=0.4* klam*np.log(10)*e_ebv
        e_flux_dered = ( (e_flux/flux)**2 + s_f**2)**0.5 * flux_dered
        return flux_dered, e_flux_dered


