# !usr/bin/env python
# -*- coding: utf-8 -*-
# F. Belfiore, 06/06/2018
#   Adapted from DAP's setup.py from K. Westfall

# Imports
import os
import glob
from setuptools import setup, find_packages


_VERSION = '1.0.0'
_RELEASE = 'dev' not in _VERSION
#_MAX_PYTHON_VERSION = '2.9'


def get_data_files():
    """Generate the list of data files."""
    data_files = []
    data_roots = [ 'grids' ]
    for root in data_roots:
        for path, directories, files in os.walk(root):
            for f in files:
                data_path = '/'.join(path.split('/')[1:])
                data_files.append(os.path.join(data_path, f))
    return data_files


#def get_scripts():
#    """ Grab all the scripts in the bin directory.  """
#    scripts = []
#    if os.path.isdir('bin'):
#        scripts = [ fname for fname in glob.glob(os.path.join('bin', '*'))
#                                if not os.path.basename(fname).endswith('.rst') and
#                                   not os.path.basename(fname).endswith('.bash') ]
#    return scripts


def get_requirements():
    """ Get the package requirements from the system file. """
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    install_requires = [line.strip().replace('==', '>=') for line in open(requirements_file)
                        if not line.strip().startswith('#') and line.strip() != '']
#    install_requires = [line.strip() for line in open(requirements_file)
#                        if not line.strip().startswith('#') and line.strip() != '']
    return install_requires


def run_setup(data_files,  packages, install_requires):

    setup(name='izi',
          version=_VERSION,
          license='BSD3',
          description='IZI',
          long_description=open('README.md').read(),
          author='Francesco Belfiore',
          author_email='fbelfior@ucsc.edu',
          keywords='astronomy bayesian metallicity',
          url='https://github.com/sdss/izi',
          #python_requires='>=2.6,<3.0',
          packages=packages,
          package_data={'': data_files},
          include_package_data=True,
          install_requires=install_requires,
          classifiers=[
              'Development Status :: 4 - Beta',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Natural Language :: English',
              'Operating System :: OS Independent',
              'Programming Language :: Python',
              'Topic :: Documentation :: Sphinx',
              'Topic :: Scientific/Engineering :: Astronomy',
              'Topic :: Software Development :: Libraries :: Python Modules'
          ],
          )

#-----------------------------------------------------------------------
if __name__ == '__main__':

    # Compile the data files to include
    data_files = get_data_files()

    # Compile the scripts in the bin/ directory
#    scripts = get_scripts()

    # Get the packages to include
    packages = find_packages()

    # Collate the dependencies based on the system text file
    install_requires = get_requirements()

    # Run setup from setuptools
    run_setup(data_files, packages, install_requires)

