# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import configparser

import numpy as np


class Options(object):
    
    def __init__(self, configfile='settings.cfg'):
        """
        Options (opt) contains all of the settings from the configuration file.
        
        The file is read in, compared with a dictionary of defaults, type and
        appropriate values are enforced, and derived settings are calculated.
        
        Parameters
        ----------
        configfile : str
            Name of configuration file to read.
        
        """
        
        self.configfile = configfile
        
        # !!! Currently same order as settings.cfg; both need to be reorganized
        # Define defaults in a dictionary
        defaults = { 
            
            # RUN PARAMETERS
            'title'          : 'REDPy Catalog',
            'outputPath'     : '',
            'groupName'      : 'default',
            'filename'       : 'redpytable.h5',
            'minorph'        : 0.05,
            'maxorph'        : 7.0,
            'nsec'           : 3600,
            
            # STATION PARAMETERS
            'nsta'           : 8,
            'station'        : 'SEP,YEL,HSR,SHW,EDM,STD,JUN,SOS',
            'channel'        : 'EHZ,EHZ,EHZ,EHZ,EHZ,EHZ,EHZ,EHZ',
            'network'        : 'UW,UW,UW,UW,UW,UW,UW,UW',
            'location'       : '--,--,--,--,--,--,--,--',
            'samprate'       : 100.,
            'fmin'           : 1.,
            'fmax'           : 10.,
            'filomin'        : 1.,
            'filomax'        : 2.5,
            'fiupmin'        : 5.,
            'fiupmax'        : 10.,
            'fispanlow'      : -0.5,
            'fispanhigh'     : 0.5,
            
            # DATA SOURCE
            'server'         : 'IRIS',
            'port'           : 16017,
            'searchdir'      : './',
            'filepattern'    : '*',
            'preload'        : 10.,
            
            # TRIGGERING SETTINGS
            'trigalg'        : 'classicstalta',
            'nstaC'          : 5,
            'lwin'           : 8.,  # 7
            'swin'           : 0.7, # 0.8
            'trigon'         : 3.,
            'trigoff'        : 2.,
            'offset'         : '0',
            
            # CROSS-CORRELATION PARAMETERS
            'winlen'         : 1024,
            'cmin'           : 0.7,
            'ncor'           : 4,
            'merge_percent'  : 0.,
            'corr_nrecent'   : 0,
            'corr_nlargest'  : 0,
            
            # PLOTTING PARAMETERS
            'plotformat'     : 'eqrate,fi,occurrence+occurrencefi,longevity',
            'printsta'       : 2,
            'minplot'        : 5,
            'dybin'          : 1.,
            'hrbin'          : 1.,
            'occurbin'       : 1., # convert to decimal days
            'recbin'         : 1., # convert to decimal days
            'fixedheight'    : False,
            'recplot'        : 14.,
            'mminplot'       : 0,
            'mhrbin'         : 1.,
            'mrecbin'        : 1., # convert to decimal days
            'mrecplot'       : 30.,
            'verbosecatalog' : False,
            'anotfile'       : '',
            'amplims'        : 'global', # enforce else 'family'
            
            # CHECK COMCAT (/EXTERNAL) CATALOG
            'checkComCat'    : False,
            'stalats'        : '46.200210,46.209550,46.174280,' + \
                '46.193470,46.197170,46.237610,46.147060,46.243860',
            'stalons'        : '-122.190600,-122.188990,-122.180650,' + \
                '-122.236350,-122.151210,-122.223960,-122.152430,-122.137870',
            'serr'           : 5.,
            'locdeg'         : 0.5,
            'regdeg'         : 2.,
            'regmag'         : 2.5,
            'telemag'        : 4.5,
            'matchMax'       : 0,
            
            # AUTOMATED SPIKE AND TELESEISM REMOVAL
            'kurtwin'        : 5.,
            'kurtmax'        : 80.,
            'kurtfmax'       : 150.,
            'oratiomax'      : 0.15,
            'telefi'         : -1.,
            'teleok'         : 2 # 1
            
        }
        
        # Read from configfile
        config = configparser.ConfigParser()
        config.read(self.configfile)
        
        # Assign values, enforcing type from defaults
        for key in defaults:
            if config.has_option('Settings',key):
                if type(defaults[key]) == int:
                    setattr(self, key, config.getint('Settings',key))
                elif type(defaults[key]) == float:
                    setattr(self, key, config.getfloat('Settings',key))
                elif type(defaults[key]) == bool:
                    setattr(self, key, config.getboolean('Settings',key))
                else: # Last option is string
                    setattr(self, key, config.get('Settings',key))
            else:
                setattr(self, key, defaults[key])
        
        # Do any conversions necessary
        for key in ['occurbin', 'recbin', 'mrecbin']:
            setattr(self, key, getattr(self, key)/24)
        
        # Check for any renamed parameters to allow old config files to be used
        # !!! None so far
        
        # Enforce parameters to make sense
        for key in ['station', 'channel', 'network', 'location']:
            if len(getattr(self, key).split(',')) != self.nsta:
                raise ValueError(
                    f'{key} length and nsta mismatch, check {configfile}')
        for key in ['nstaC', 'ncor', 'teleok']:
            if getattr(self,key) > self.nsta:
                raise ValueError(
                    f'{key} larger than nsta, check {configfile}')
        if self.printsta >= self.nsta:
            raise ValueError(
                f'printsta must be 0-{self.nsta-1}, check {configfile}')
        for key in ['filomin', 'filomax', 'fiupmin', 'fiupmax']:
            if (getattr(self,key) < self.fmin) or \
               (getattr(self,key) > self.fmax):
                raise ValueError(
                    f'{key} not within filter passband, check {configfile}')
        for key in ['fmax', 'fmin']:
            if (getattr(self,key) > self.samprate/2):
                raise ValueError(
                    f'{key} above Nyquist, check {configfile}')
        if (self.amplims != 'global') and (self.amplims != 'family'):
            raise ValueError(
                f'Use either global or family for amplims, check {configfile}')
        
        # Define derived parameters
        self.ptrig = 1.5*self.winlen/self.samprate
        self.atrig = 3*self.winlen/self.samprate
        self.mintrig = 0.75*self.winlen/self.samprate
        self.wshape = int((self.ptrig + self.atrig)*self.samprate) + 1
        self.maxdt = np.max(np.fromstring(self.offset, sep=','))
