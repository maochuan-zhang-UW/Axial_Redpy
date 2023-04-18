# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling functions related to creating maps.

The primary function of this module is to support the .output() method of
Detector() objects. The .output() method generates various images and .html
files so the user may easily browse and export the contents of REDPy's
detections.
"""
import os

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.transforms import offset_copy
from obspy.geodetics import kilometers2degrees


def create_local_map(detector, fnum, local):
    """
    Make map centered on local matches from external catalog.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    local : dict
        Location information of local events to be plotted.
    fnum : int
        Family number.

    """
    if len(local['deps']) > 0:
        # pylint: disable=E0110
        # Abstract class instantiation from cartopy
        plate_carree = ccrs.PlateCarree()
        # pylint: enable=E0110
        stamen_terrain = cimgt.StamenTerrain()
        extent = [np.median(local['lons']) - detector.get('locdeg')/2,
                  np.median(local['lons']) + detector.get('locdeg')/2,
                  np.median(local['lats']) - detector.get('locdeg')/4,
                  np.median(local['lats']) + detector.get('locdeg')/4]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
        ax.set_extent(extent, crs=plate_carree)
        ax.add_image(stamen_terrain, 11)
        ax.set_xticks(np.arange(np.floor(10*(extent[0]))/10,
                      np.ceil(10*(extent[1]))/10, 0.1), crs=plate_carree)
        ax.set_yticks(np.arange(np.floor(10*(extent[2]))/10,
                      np.ceil(10*(extent[3]))/10, 0.1), crs=plate_carree)
        ax.set_extent(extent, crs=plate_carree)  # Reset to fill image
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        plt.yticks(rotation=90, va='center')
        ax.scatter(local['lons'], local['lats'], s=20, marker='o',
                   color='white', transform=plate_carree)
        ax.scatter(local['lons'], local['lats'], s=5, marker='o',
                   color='red', transform=plate_carree)
        ax.scatter(detector.get('stalons'), detector.get('stalats'),
                   marker='^', color='k', facecolors='None',
                   transform=plate_carree)
        scalebar_lat = 0.05*(detector.get('locdeg')/2) + extent[2]
        scalebar_lon_left = 0.05*(detector.get('locdeg')) + extent[0]
        scalebar_len = kilometers2degrees(10) / np.cos(
            scalebar_lat*np.pi / 180)
        ax.plot((scalebar_lon_left, scalebar_lon_left + scalebar_len),
                (scalebar_lat, scalebar_lat), 'k-',
                transform=plate_carree, lw=2)
        # pylint: disable=W0212
        # I trust use of protected member here.
        geodetic_transform = plate_carree._as_mpl_transform(ax)
        # pylint: enable=W0212
        text_transform = offset_copy(geodetic_transform, units='dots', y=5)
        ax.text(scalebar_lon_left + scalebar_len/2,
                scalebar_lat, '10 km', ha='center', transform=text_transform)
        plt.title(f'{len(local["deps"])} potential local matches '
                  f'(~{np.mean(local["deps"]):3.1f} km depth)')
        plt.tight_layout()
        plt.savefig(os.path.join(detector.get('output_folder'), 'families',
                    f'map{fnum}.png'), dpi=100)
        plt.close()
