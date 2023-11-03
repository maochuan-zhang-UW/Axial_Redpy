<img src="https://code.usgs.gov/vsc/REDPy/-/raw/main/img/logo.png"
width=800 alt="REDPy Logo" />

REDPy (Repeating Earthquake Detector in Python) is a tool for automated
detection and analysis of repeating earthquakes in continuous data. It works
without any previous assumptions of what repeating seismicity looks like (that
is, does not require a template event). Repeating earthquakes are clustered
into "families" based on waveform similarity via cross-correlation across
multiple stations. All data, including waveforms, are stored in a HDF5 database
using [PyTables](http://www.pytables.org/).

Author: Alicia Hotovec-Ellis, U.S. Geological Survey | ahotovec-ellis@usgs.gov

## Installation

Download the [latest release](https://code.usgs.gov/vsc/REDPy/-/releases)
or use `git` to clone the entire repository to a working directory. You may
either run REDPy in this directory (a default directory structure has been
created here to hold your `.h5` files and runs), or you can configure your own
directory structure.

REDPy runs on Python 3.9+, with the following major package dependencies:  
[numpy](http://www.numpy.org/) | [scipy](http://www.scipy.org/) | 
[matplotlib](http://www.matplotlib.org/) | [obspy](http://www.obspy.org/) | 
[pytables](http://www.pytables.org/) | [pandas](http://pandas.pydata.org/) | 
[scikit-learn](http://scikit-learn.org/) | [bokeh](http://bokeh.pydata.org/) | 
[cartopy](http://scitools.org.uk/cartopy/)

These dependencies can be installed via [Anaconda](https://www.anaconda.com/download)
on the command line. It is *highly* recommended to use a virtual environment
so that your REDPy environment does not conflict with any other Python packages
you may be using. This can be done with the following single command:
```
conda env create -f environment.yml
```
This creates an environment named `redpy` with Python 3.11 that should mirror
the environment used to develop the code. If you wish to use a different
version of Python 3.9+, with the most up-to-date versions of the dependencies,
you can also: 
```
conda config --add channels conda-forge
conda create -n redpy bokeh cartopy obspy pandas pytables scikit-learn
```
Advanced users are welcome to utilize the package and environment managers of
their choice (e.g., `pyenv`).

Once the environment has been created, dependencies have been installed, and
you are in the directory that contains `pyproject.toml`, activate the
environment and run `pip` to install the `redpy` package and the scripts:
```
conda activate redpy
pip install .
```
All of the console scripts start with `redpy-` and you can use the `-h` flag
to get information on their usage.

## Example Usage

Once REDPy is installed and you are in the `redpy` environment, REDPy can be
run out of the box with the following commands to test if the code is working
on your computer. If it completes without error, it will produce files in a
folder `./runs/default/` after several minutes (~5, but depends on your
download speed).
```
redpy-initialize -v -c settings.cfg
redpy-catfill -v -a -f -c settings.cfg example_catalog.csv
redpy-backfill -v -c settings.cfg -s 2004-09-23 -e 2004-09-24
```

## Documentation

Check out the [Wiki](https://code.usgs.gov/vsc/REDPy/-/wikis/home)
for more detailed usage and documentation, including a comprehensive overview
of the various outputs REDPy generates.

## Citation

If you would like to reference REDPy in your paper, please cite the following
abstract until the *Electronic Seismologist* paper is published:

> Hotovec-Ellis, A.J., and Jeffries, C. (2016) Near Real-time Detection,
Clustering, and Analysis of Repeating Earthquakes: Application to Mount St.
Helens and Redoubt Volcanoes – *Invited*, presented at Seismological Society
of America Annual Meeting, Reno, Nevada, 20 Apr.

The code may also be cited directly as:

> Hotovec-Ellis, A.J. (2023) REDPy – Repeating Earthquake Detector in
Python (Version 1.0.0), U.S. Geological Survey Software Release,
https://doi.org/10.5066/P94I4HRI.

## License and Disclaimer
[License](https://code.usgs.gov/vsc/REDPy/-/blob/main/LICENSE.md):
This project is in the public domain.

[Disclaimer](https://code.usgs.gov/vsc/REDPy/-/blob/main/DISCLAIMER.md):
This software is preliminary or provisional and is subject to revision.