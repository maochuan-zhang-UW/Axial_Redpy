# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
REDPy is the Repeating Earthquake Detector in Python.

This module organizes detections of earthquakes from continuous waveform
data into groups of 'families' based on their waveform similarity. This was
written primarily to support volcano monitoring efforts, where highly
similar earthquakes ('repeaters') often occur in both repose and unrest.

# !!! Documentation and reference here!
"""
from importlib.metadata import version

from redpy.config import Config
from redpy.detector import Detector
from redpy.scripts import *
from redpy.table import Table
from redpy.trigger import Trigger
from redpy.waveform import Waveform

__version__ = version('redpy')
