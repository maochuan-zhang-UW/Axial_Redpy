<img src="https://raw.githubusercontent.com/ahotovec/REDPy/master/img/logo.png" width=800 alt="REDPy Logo" />

## Overview
REDPy (Repeating Earthquake Detector in Python) is a tool for automated detection and analysis of repeating earthquakes in continuous data. It works without any previous assumptions of what repeating seismicity looks like (that is, does not require a template event). Repeating earthquakes are clustered into "families" based on cross-correlation across multiple stations. All data, including waveforms, are stored in an HDF5 table using [PyTables](http://www.pytables.org/).

## Installation
Download the [latest release](https://github.com/ahotovec/REDPy/archive/master.zip) or use `git` to clone the entire repository to a working directory. You may either run REDPy in this directory (I have set up a default directory structure to hold your .h5 files and runs), or you can configure your own directory structure.

REDPy runs on Python 3.9+, with the following major package dependencies:  
[numpy](http://www.numpy.org/) | [scipy](http://www.scipy.org/) | [matplotlib](http://www.matplotlib.org/) | [obspy](http://www.obspy.org/) | [pytables](http://www.pytables.org/) | [pandas](http://pandas.pydata.org/) | [scikit-learn](http://scikit-learn.org/) | [bokeh](http://bokeh.pydata.org/) | [cartopy](http://scitools.org.uk/cartopy/)

These dependencies can be installed via [Anaconda](https://www.anaconda.com/download) on the command line. I *highly* recommend using a virtual environment so that your REDPy environment does not conflict with any other Python packages you may be using. This can be done with the following command:
```
conda env create -f environment.yml
```
This creates an environment named `redpy` with Python 3.11 that should mirror the environment I used to develop the code. If you wish to use a different version of Python 3.9+, with the most up-to-date versions of the dependencies, you can also: 
```
conda config --add channels conda-forge
conda create -n redpy bokeh cartopy obspy pandas pytables scikit-learn
```
Advanced users are welcome to utilize the package and environment managers of their choice (e.g., `pyenv`).

Once the environment has been created, dependencies have been installed, and you are in the directory that contains `pyproject.toml`, activate the environment and run `pip` to install the `redpy` package and the scripts:
```
conda activate redpy
pip install .
```
All of the console scripts start with `redpy-` and you can use the `-h` flag to get information on their usage.


## Example Usage
Once REDPy is installed and you are in the `redpy` environment, REDPy can be run out of the box with the following commands to test if the code is working on your computer. If it completes without error, it will produce files in a folder `./runs/default/` after several minutes (~5, but depends on your download speed).
```
redpy-initialize -v -c settings.cfg
redpy-catfill -v -a -f -c settings.cfg example_catalog.csv
redpy-backfill -v -c settings.cfg -s 2004-09-23 -e 2004-09-24
```

Check out the [Documentation](https://github.com/ahotovec/REDPy/wiki) for more detailed usage!

## Reference

If you would like to reference REDPy in your paper, please cite the following abstract until I finish writing the *Electronic Seismologist* paper for it:

> Hotovec-Ellis, A.J., and Jeffries, C., 2016. Near Real-time Detection, Clustering, and Analysis of Repeating Earthquakes: Application to Mount St. Helens and Redoubt Volcanoes – *Invited*, presented at Seismological Society of America Annual Meeting, Reno, Nevada, 20 Apr.
