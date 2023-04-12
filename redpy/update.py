# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling functions related to updating the catalog of detections.

The primary function of this module is to support the .update() method of
Detector() objects. The .update() method creates triggers from continuous
waveform data that are cross-correlated and sorted into families.
"""
