"""
Module for handling functions related to creating maps.

The primary function of this module is to support the .output() method of
Detector() objects. The .output() method generates various images and .html
files so the user may easily browse and export the contents of REDPy's
detections.
"""
import os

import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.transforms import offset_copy
from obspy.geodetics import kilometers2degrees

matplotlib.use('agg')


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
        plate_carree = ccrs.PlateCarree()
        cartopy.config['cache_dir'] = os.path.join('.', '.cache')
        background_tile = cimgt.GoogleTiles(cache=True, url=(
            'https://server.arcgisonline.com/ArcGIS/rest/services/'
            'World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg'))
        extent = [np.median(local['lons']) - detector.get('locdeg')/2,
                  np.median(local['lons']) + detector.get('locdeg')/2,
                  np.median(local['lats']) - detector.get('locdeg')/4,
                  np.median(local['lats']) + detector.get('locdeg')/4]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=background_tile.crs)
        ax.set_extent(extent, crs=plate_carree)
        ax.add_image(background_tile, 11)
        ax.set_xticks(np.arange(np.floor(10*(extent[0]))/10,
                      np.ceil(10*(extent[1]))/10, 0.1), crs=plate_carree)
        ax.set_yticks(np.arange(np.floor(10*(extent[2]))/10,
                      np.ceil(10*(extent[3]))/10, 0.1), crs=plate_carree)
        ax.set_extent(extent, crs=plate_carree)  # Reset to fill image
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        plt.yticks(rotation=90, va='center')
        ax.scatter(detector.get('stalons'), detector.get('stalats'),
                   marker='^', color='k', facecolors='None', linewidths=0.5,
                   transform=plate_carree)
        ax.scatter(local['lons'], local['lats'], s=25, marker='o',
                   color='w', transform=plate_carree)
        ax.scatter(local['lons'], local['lats'], s=10, marker='o',
                   linewidths=0.5, edgecolors='#7e0729',
                   color='#f34535', transform=plate_carree)
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


def get_tiles(detector):
    """
    Ensure background tiles are downloaded to local cache and desaturated.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.

    """
    if not os.path.exists('.cache'):
        os.mkdir(os.path.join('.', '.cache'))
        os.mkdir(os.path.join('.', '.cache', 'GoogleTiles'))
    plate_carree = ccrs.PlateCarree()
    cartopy.config['cache_dir'] = os.path.join('.', '.cache')
    background_tile = cimgt.GoogleTiles(cache=True, url=(
        'https://server.arcgisonline.com/ArcGIS/rest/services/'
        'World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg'))
    extent = [np.median(detector.get('stalons')) - 1.5*detector.get('locdeg'),
              np.median(detector.get('stalons')) + 1.5*detector.get('locdeg'),
              np.median(detector.get('stalats')) - 1.5*detector.get('locdeg'),
              np.median(detector.get('stalats')) + 1.5*detector.get('locdeg')]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=background_tile.crs)
    ax.set_extent(extent, crs=plate_carree)
    ax.add_image(background_tile, 11)
    plt.savefig(os.path.join('.', '.cache', 'tmp.png'))
    plt.close()
