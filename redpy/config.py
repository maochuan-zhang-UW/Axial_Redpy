"""
Module for handling REDPy Config() object.

The Config() object contains all of the settings used to define a single
REDPy run. Settings are explained in detail in the original 'settings.cfg'
file provided with the source code, which is set to the default settings.
"""
import configparser
from copy import deepcopy

import numpy as np


class Config():
    """
    Container for all configuration parameters describing a single run.

    Attributes
    ----------
    configfile : str
        Name of read configuration file.
    custom_settings : list
        List of keys in settings that are different from the default.
    settings : dict
        Dictionary of all settings, populated either from defaults or
        the configuration file.
    verbose : bool
        Enable additional print statements.
    """

    def __init__(self, configfile='settings.cfg', verbose=False):
        """
        Read settings from the configuration file or apply defaults.

        The configuration file is read in, compared with a dictionary of
        defaults, type and appropriate values are enforced, and derived
        settings are calculated.

        Parameters
        ----------
        configfile : str, optional
            Name of configuration file to read.
        verbose : bool, optional
            Enable additional print statements.

        """
        self.configfile = configfile
        self.verbose = verbose
        self.settings = {

            #  Name / Path Parameters  #
            'title': 'REDPy Catalog',
            'outputpath': './runs/',
            'groupname': 'default',
            'filename': './h5/redpytable.h5',

            #  Data Source Parameters  #
            'server': 'IRIS',
            'searchdir': './',
            'filepattern': '*',
            'preload': 10.,

            #  SCNL Paramters  #
            'nsta': 8,
            'station': 'SEP,YEL,HSR,SHW,EDM,STD,JUN,SOS',
            'channel': 'EHZ,EHZ,EHZ,EHZ,EHZ,EHZ,EHZ,EHZ',
            'network': 'UW,UW,UW,UW,UW,UW,UW,UW',
            'location': '--,--,--,--,--,--,--,--',
            'offset': '0',

            #  Filtering Parameters  #
            'samprate': 100.,
            'fmin': 1.,
            'fmax': 10.,

            #  Frequency Index Parameters  #
            'filomin': 1.,
            'filomax': 2.5,
            'fiupmin': 5.,
            'fiupmax': 10.,

            #  Triggering Parameters  #
            'trigalg': 'classicstalta',
            'nstac': 5,
            'swin': 0.7,  # 0.8
            'lwin': 8.,  # 7
            'trigon': 3.,
            'trigoff': 2.,

            #  Quality Control Parameters  #
            'kurtwin': 5.,
            'kurtmax': 80.,
            'kurtfmax': 150.,
            'oratiomax': 0.20,  # 0.15
            'telefi': -1.5,  # -1
            'teleok': 2,  # 1

            #  Correlation Parameters  #
            'winlen': 1024,
            'cmin': 0.7,
            'ncor': 4,
            'use_nthcor': False,
            'corr_permit': 0.05,  # 0.1? 0.15?
            'corr_nrecent': 0,  # 25?
            'corr_nyoungest': 0,  # 25?
            'corr_nlargest': 0,  # 50?

            #  Run Parameters  #
            'minorph': 0.05,
            'maxorph': 7.0,
            'nsec': 3600,
            'max_famlen': 30000,  # 1000000
            'merge_ratio': 0.,  # 0.6?
            'always_verbose': False,

            #  Timeline Parameters  #
            'plotformat': 'eqrate,fi,occurrence+occurrencefi,longevity',
            'bokehendtime': 'trigger',  # enforce else 'now'
            'timeline_vs': 'orphans',  # enforce else 'triggers'
            'minplot': 5,
            'mminplot': 0,
            'recplot': 14.,
            'mrecplot': 30.,
            'dybin': 1.,
            'hrbin': 1.,
            'mhrbin': 1.,
            'occurbin': 1.,
            'recbin': 1.,  # convert to decimal days
            'mrecbin': 1.,  # convert to decimal days
            'fixedheight': False,
            'fispanlow': -0.5,
            'fispanhigh': 0.5,
            'anotfile': '',

            #  Family Plot Parameters  #
            'printsta': 2,
            'amplims': 'global',  # enforce else 'family'

            #  Text Catalog Parameters  #
            'verbosecatalog': False,

            #  External Catalog Parameters  #
            'checkcomcat': False,
            'datacenter': 'USGS',
            'stalats': ('46.200210,46.209550,46.174280,46.193470,46.197170,'
                        '46.237610,46.147060,46.243860'),
            'stalons': ('-122.190600,-122.188990,-122.180650,-122.236350,'
                        '-122.151210,-122.223960,-122.152430,-122.137870'),
            'serr': 5.,
            'locdeg': 0.5,
            'regdeg': 2.,
            'regmag': 2.5,
            'telemag': 4.5,
            'matchmax': 0,

        }
        self.custom_settings = []
        self._update_from_cfg()
        self._convert_to_days()
        self._populate_derived()
        if self.verbose:
            print(self)

    def __repr__(self):
        """Format representation string."""
        return self.append_custom(f'redpy.Config(configfile='
                                  f'"{self.configfile}")')

    def __str__(self):
        """Format print string."""
        return self.append_custom(f'Using config file: "{self.configfile}"')

    def append_custom(self, string):
        """
        Append a formatted ending to a string listing the custom settings.

        If no custom settings exist, appends with message that all settings
        are default.

        Parameters
        ----------
        string : str
            A string, such as for building repr(), to append.

        Returns
        -------
        str
            Modified string, with custom settings appended.

        """
        if self.custom_settings:
            string += ' with custom settings:'
            for key in self.custom_settings:
                string += f'\n  {key}={self.get(key)}'
        else:
            string += ' with all default settings'
        return string

    def copy(self):
        """Return a deepcopy that can be safely altered."""
        return deepcopy(self)

    def get(self, key):
        """Return the keyed attribute's value."""
        if key in ('configfile', 'verbose', 'settings', 'custom_settings'):
            return getattr(self, key)
        return self.settings[key]

    def set(self, key, value):
        """Set the keyed attribute's value in the proper place."""
        if key in ('configfile', 'verbose', 'settings', 'custom_settings'):
            setattr(self, key, value)
        else:
            self.settings[key] = value

    def _convert_to_days(self):
        """Convert keys given in hours to decimal days."""
        for key in ['recbin', 'mrecbin']:
            self.set(key, self.get(key)/24)

    def _enforce(self):
        """Enforce parameters to make sense."""
        for key in ['station', 'channel', 'network', 'location']:
            self.set(key, self.get(key).split(','))
            if len(self.get(key)) != self.get('nsta'):
                raise ValueError(
                    f'{key} and nsta mismatch, check {self.configfile}')
        for key in ['nstac', 'ncor', 'teleok']:
            if self.get(key) > self.get('nsta'):
                raise ValueError(
                    f'{key} larger than nsta, check {self.configfile}')
        if self.get('printsta') >= self.get('nsta'):
            raise ValueError(
                f'printsta must be 0-{self.get("nsta")-1}, check '
                f'{self.configfile}')
        for key in ['filomin', 'filomax', 'fiupmin', 'fiupmax']:
            if (self.get(key) < self.get('fmin')) or (
                    self.get(key) > self.get('fmax')):
                raise ValueError(
                    f'{key} not within filter passband, check '
                    f'{self.configfile}')
        for key in ['fmax', 'fmin']:
            if self.get(key) > self.get('samprate')/2:
                raise ValueError(
                    f'{key} above Nyquist, check {self.configfile}')
        if self.get('amplims') not in ('global', 'family'):
            raise ValueError(
                f'Use either global or family for amplims, check '
                f'{self.configfile}')

    def _populate_derived(self):
        """Populate derived settings."""
        samprate = self.get('samprate')
        self.set('ptrig', 1.5*self.get('winlen')/samprate)
        self.set('start_sample', 1.5*self.get('winlen'))
        self.set('atrig', 3*self.get('winlen')/samprate)
        self.set('mintrig', 0.75*self.get('winlen')/samprate)
        self.set(
            'wshape', int((self.get('ptrig') + self.get('atrig'))*samprate)+1)
        self.set(
            'maxdt', np.max(np.fromstring(self.get('offset'), sep=',')))
        self.set('cperm', self.get('cmin')-self.get('corr_permit'))
        self.set('output_folder',
                 f"{self.get('outputpath')}{self.get('groupname')}")
        self.set('stalats', np.array(
            self.get('stalats').split(',')).astype(float))
        self.set('stalons', np.array(
            self.get('stalons').split(',')).astype(float))
        self.set('latitude_center', np.mean(self.get('stalats')))
        self.set('longitude_center', np.mean(self.get('stalons')))

    def _update_from_cfg(self):
        """Assign values from configuration file, enforcing type."""
        parser = configparser.ConfigParser()
        parser.read(self.configfile)
        for item in list(parser.items('Settings')):
            if item[0] in self.settings:
                key = item[0]
                if isinstance(self.get(key), bool):
                    value = parser.getboolean('Settings', key)
                elif isinstance(self.get(key), float):
                    value = parser.getfloat('Settings', key)
                elif isinstance(self.get(key), int):
                    value = parser.getint('Settings', key)
                else:
                    value = parser.get('Settings', key)
                if value != self.get(key):
                    self.set(key, value)
                    self.custom_settings.append(key)
                if key == 'always_verbose' and value:
                    self.set('verbose', True)
            else:
                raise ValueError(
                    f'Unrecognized or deprecated setting "{item[0]}"; '
                    f'aborting! Please check {self.configfile}.')
        self._enforce()
