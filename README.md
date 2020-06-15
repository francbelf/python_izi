# izi

A python version of Gulliermo Blanc's IZI (Inferring the Gas Phase Metallicity (Z) and Ionization Parameter (q) of Ionized Nebulae Using Bayesian Statistics) code to calculate metallicity and ionization parameter in a Bayesian fashion by comparing emission line fluxes with photoionization models. 
A new version of the code is also provided which fits metallicity, ionization parameter and dust extinction (assuming a Calzetti law) simultaneously (called izi_mcmc). This version follows the same logic as the original IZI code, but uses the package emcee to sample the parameter space spanned by the three parameters.

Several model grids are included in the current installation.

## Installation

First clone the full repo by or clicking the download button on GitHub.

`git clone --recursive https://github.com/sdss/izi.git`

then do

`python -m pip install -e .`

## Usage
To use the direct python translation of Blanc's 2015 IDL izi do

`from izi import izi`

or try the examples/test_izi script for a simple example.

To use the new MCMC version of python IZI which also fits for extinction do

`from izi_MCMC import izi_MCMC`

Try out examples/test_MCMC for an example. The example also explains how to make use of the prior on the ionisation parameter described in Mingozzi et al., 2020.

## Citation

If you are using this code please cite the both papers below.
The original paper describing the IZI code:

Blanc et al., 2015, ApJ, 798, 99 (https://ui.adsabs.harvard.edu/abs/2015ApJ...798...99B/abstract)

AND the paper describing the current python version of the code and the mcmc extension which includes fitting for the extinction:

Mingozzi et al., 2020, A&A, 636, A42 (https://ui.adsabs.harvard.edu/abs/2020A%26A...636A..42M/abstract)

## Problems?

Please contact Matilde Mingozzi (matilde.mingozzi@inaf.it) and/or Francesco Belfiore (francesco.belfiore@inaf.it).
