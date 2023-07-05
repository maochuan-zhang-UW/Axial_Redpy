# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
This script is necessary to install the 'redpy' module to environment as well
as all of the command line scripts. See README.md for full installation
instructions.
"""
from setuptools import setup, find_packages

# Command line scripts
ENTRY_POINTS = {
    'console_scripts': [
        'redpy-backfill = redpy.scripts.backfill:main',
        'redpy-catfill = redpy.scripts.catfill:main',
        'redpy-clear-junk = redpy.scripts.clearjunk:main',
        'redpy-compare-catalog = redpy.scripts.comparecatalog:main',
        'redpy-create-pdf-family = redpy.scripts.createpdffamily:main',
        'redpy-create-pdf-overview = redpy.scripts.createpdfoverview:main',
        'redpy-create-report = redpy.scripts.createreport:main',
        'redpy-distant-families = redpy.scripts.distantfamilies:main',
        'redpy-extend-table = redpy.scripts.extendtable:main',
        'redpy-force-plot = redpy.scripts.forceplot:main',
        'redpy-initialize = redpy.scripts.initialize:main',
        'redpy-make-meta = redpy.scripts.makemeta:main',
        'redpy-plot-junk = redpy.scripts.plotjunk:main',
        'redpy-remove-family = redpy.scripts.removefamily:main',
        'redpy-remove-family-gui = redpy.scripts.removefamilygui:main',
        'redpy-remove-small-family = redpy.scripts.removesmallfamily:main',
        ('redpy-write-family-locations = '
         'redpy.scripts.writefamilylocations:main')
        ]
    }

setup(
    name='redpy',
    version='1.0.0',
    author='Alicia Hotovec-Ellis',
    description='Repeating Earthquake Detector in Python',
    long_description="""
        REDPy (Repeating Earthquake Detector in Python) is a tool for
        automated detection and analysis of repeating earthquakes in
        continuous data. It works without any previous assumptions of what
        repeating seismicity looks like (that is, does not require a
        template event). Repeating earthquakes are clustered into "families"
        based on cross-correlation across multiple stations. All data,
        including waveforms, are stored in an HDF5 table using PyTables.
    """,
    url='https://github.com/ahotovec/REDPy',  # !!!
    keywords='seismology, earthquake, multiplet',
    python_requires='>=3.9, <4',
    packages=find_packages(
        include=['redpy', 'redpy.*', 'redpy.outputs.*', 'redpy.scripts.*']),
    install_requires=[
        'bokeh>=3.0',
        'cartopy',
        'matplotlib',
        'numpy',
        'obspy>=1.4.0',
        'pandas',
        'tables',
        'scikit-learn',
        'scipy'
    ],
    entry_points = ENTRY_POINTS,
)
