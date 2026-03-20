"""
Module for handling functions related to creating maps.

The primary function of this module is to support the .output() method of
Detector() objects. The .output() method generates various images and .html
files so the user may easily browse and export the contents of REDPy's
detections.
"""
import os

import scipy.io as sio

import redpy.correlation
import cartopy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.transforms import offset_copy
from obspy import UTCDateTime
from obspy.geodetics import kilometers2degrees
from obspy.imaging.beachball import beach

matplotlib.use('agg')

# Axial caldera rim coordinates (from Chadwick) — open polyline, do not close
_CALDERA_RIM_LON = [
    -130.004785563058, -130.010476202888, -130.018881564079,
    -130.023946125193, -130.028718653506, -130.03045121938,
    -130.03067948565,  -130.031733279709, -130.0314446535,
    -130.036188782208, -130.036950110789, -130.039953347,
    -130.038595675479, -130.035927416999, -130.018067675296,
    -130.013629193751, -130.010365710979, -130.008647442296,
    -130.007262470669, -130.006042022411, -130.00517862949,
    -130.001868199523, -130.001154359192, -130.000949059432,
    -129.99939353433,  -129.997797388662, -129.995357566829,
    -129.993956176267, -129.993678708114, -129.993140494256,
    -129.992087550788, -129.991186410747, -129.989604036931,
    -129.989238986151, -129.989728217453, -129.98548409867,
    -129.98478812249]
_CALDERA_RIM_LAT = [
    45.9207755734405, 45.9238241104543, 45.9351908809594,
    45.9412238501725, 45.949881200114,  45.9511765797916,
    45.9542732243167, 45.9558130656063, 45.9586760104296,
    45.9656647517656, 45.9698291665232, 45.9750458167927,
    45.9847117727418, 45.9883113986506, 45.993358288674,
    45.993755284135,  45.9929499241491, 45.9924883829037,
    45.9915471582374, 45.9902469280907, 45.989777805361,
    45.9863506519894, 45.9846883853932, 45.9827833001814,
    45.9818434725493, 45.9786395525337, 45.9760388622191,
    45.9741441737512, 45.9681875631427, 45.9667620754035,
    45.9652218741086, 45.9626077204113, 45.960118640732,
    45.9588108137369, 45.9574955894078, 45.9494279735802,
    45.9487188881587]

# Station coordinates from axial_stationsNewOrder.m
_STATIONS = {
    'AXCC1': (-130.0089, 45.95468),
    'AXEC1': (-129.9797, 45.94958),
    'AXEC2': (-129.9738, 45.93967),
    'AXEC3': (-129.9785, 45.93607),
    'AXAS1': (-129.9992, 45.93356),
    'AXAS2': (-130.0141, 45.93377),
    'AXID1': (-129.978,  45.92573),
}


def _setup_map_ax(ax):
    """Apply common axis formatting for Axial map subplots."""
    ax.set_xlim(-130.05, -129.94)
    ax.set_ylim(45.91, 46.00)
    ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(labelsize=6)
    ax.set_xlabel('Longitude', fontsize=7)
    ax.set_ylabel('Latitude', fontsize=7)
    # Caldera rim (open polyline)
    ax.plot(_CALDERA_RIM_LON, _CALDERA_RIM_LAT,
            'k-', linewidth=0.8, zorder=2)
    # Stations from axial_stationsNewOrder.m
    for name, (slon, slat) in _STATIONS.items():
        ax.scatter(slon, slat, marker='^', s=30, color='k',
                   facecolors='none', linewidths=0.8, zorder=5)
        ax.annotate(name, (slon, slat), fontsize=4,
                    xytext=(2, 2), textcoords='offset points')


def _get_family_cc(detector, members):
    """
    Return per-member CC values matching the subplot_correlation plot.

    Uses the same maxrow approach as the CC subplot in fam*.png:
    builds the family sub-matrix, finds the column with the highest sum,
    and returns each member's CC against that best event.

    Parameters
    ----------
    detector : Detector object
    members : int ndarray
        Member indices into the repeaters table (time-ordered).

    Returns
    -------
    float ndarray
        CC value for each member (same length as members).
    """
    ids, ccc_sparse = detector.get_matrix()
    member_ids = ids[members]
    return redpy.correlation.subset_matrix(member_ids, ccc_sparse,
                                           return_type='maxrow')


def _scatter_with_time_cbar(fig, ax, lons, lats, times, marker_size):
    sc = ax.scatter(lons, lats, s=marker_size, c=times,
                    cmap='plasma', zorder=4, edgecolors='k',
                    linewidths=0.2, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label('Date', fontsize=7)
    n_ticks = min(5, len(times))
    tick_vals = np.linspace(times.min(), times.max(), n_ticks)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([UTCDateTime(t).strftime('%Y-%m-%d')
                         for t in tick_vals], fontsize=5)


def _scatter_with_depth_cbar(fig, ax, lons, lats, depths, marker_size):
    depth_clipped = np.clip(depths, 0, 2.5)
    sc = ax.scatter(lons, lats, s=marker_size, c=depth_clipped,
                    cmap='viridis_r', vmin=0, vmax=2.5,
                    zorder=4, edgecolors='k',
                    linewidths=0.2, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label('Depth (km)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def create_axial_map(detector, fnum, catalog_csv):
    """
    Create a cross-section location figure for a family using the DD catalog.

    Layout (all panels colored by time):
      Top-left  : map view — longitude (x) vs latitude (y)
      Top-right : side view — depth (x) vs latitude (y)  [shares y with map]
      Bottom-left: front view — longitude (x) vs depth (y) [shares x with map]
      Bottom-right: colorbar

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    fnum : int
        Family number.
    catalog_csv : str
        Path to the DD catalog CSV with columns EventID, Time, Latitude,
        Longitude, Depth, Magnitude.

    """
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec

    members = detector.get_members(fnum)
    rtimes_mpl = detector.get('rtable', 'startTimeMPL')
    event_times_sec = np.array([
        mdates.num2date(rtimes_mpl[m]).timestamp() for m in members])
    core_idx = detector.get('ftable', 'core', fnum)
    core_time_sec = mdates.num2date(rtimes_mpl[core_idx]).timestamp()

    # Load DD catalog
    try:
        cat = pd.read_csv(catalog_csv, parse_dates=['Time'])
        cat['Time'] = pd.to_datetime(cat['Time'], utc=True)
        cat_times = cat['Time'].values.astype('datetime64[ns]').astype(float) / 1e9
    except Exception:
        return

    # Match each family event to nearest catalog entry within 60s
    lats, lons, depths, matched_times = [], [], [], []
    for t_sec in event_times_sec:
        diffs = np.abs(cat_times - t_sec)
        idx = np.argmin(diffs)
        if diffs[idx] < 60:
            row = cat.iloc[idx]
            lats.append(float(row['Latitude']))
            lons.append(float(row['Longitude']))
            depths.append(float(row['Depth']))
            matched_times.append(t_sec)

    if len(lats) < 1:
        return

    lats = np.array(lats)
    lons = np.array(lons)
    depths = np.clip(np.array(depths), 0, 2.5)
    matched_times = np.array(matched_times)

    # Core event location
    core_diffs = np.abs(cat_times - core_time_sec)
    core_idx_cat = np.argmin(core_diffs)
    if core_diffs[core_idx_cat] < 60:
        core_row = cat.iloc[core_idx_cat]
        core_lon = float(core_row['Longitude'])
        core_lat = float(core_row['Latitude'])
        core_dep = float(np.clip(core_row['Depth'], 0, 2.5))
    else:
        core_lon = core_lat = core_dep = None

    # --- Shared colormap (time) ---
    t_min, t_max = matched_times.min(), matched_times.max()
    norm = mcolors.Normalize(vmin=t_min, vmax=t_max)
    cmap = cm.plasma
    MARKER_SIZE = 8

    # --- Figure layout ---
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[3, 1], height_ratios=[3, 1],
                  hspace=0.04, wspace=0.04)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_map)
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_map)
    ax_cb = fig.add_subplot(gs[1, 1])

    # --- Map (lon vs lat) ---
    ax_map.set_xlim(-130.05, -129.94)
    ax_map.set_ylim(45.91, 46.00)
    ax_map.xaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax_map.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax_map.tick_params(labelsize=6)
    ax_map.set_ylabel('Latitude', fontsize=7)
    plt.setp(ax_map.get_xticklabels(), visible=False)  # bottom panel has lon labels
    ax_map.plot(_CALDERA_RIM_LON, _CALDERA_RIM_LAT, 'k-', linewidth=0.8, zorder=2)
    for name, (slon, slat) in _STATIONS.items():
        ax_map.scatter(slon, slat, marker='^', s=30, color='k',
                       facecolors='none', linewidths=0.8, zorder=5)
        ax_map.annotate(name, (slon, slat), fontsize=4,
                        xytext=(2, 2), textcoords='offset points')

    sc = ax_map.scatter(lons, lats, s=MARKER_SIZE, c=matched_times,
                        cmap=cmap, norm=norm, zorder=4,
                        edgecolors='k', linewidths=0.2, alpha=0.85)
    if core_lon is not None:
        ax_map.scatter(core_lon, core_lat, s=50, c='red', marker='*',
                       zorder=6, edgecolors='darkred', linewidths=0.5)

    # --- Right panel (depth vs lat) ---
    ax_right.scatter(depths, lats, s=MARKER_SIZE, c=matched_times,
                     cmap=cmap, norm=norm, zorder=4,
                     edgecolors='k', linewidths=0.2, alpha=0.85)
    if core_lon is not None:
        ax_right.scatter(core_dep, core_lat, s=50, c='red', marker='*',
                         zorder=6, edgecolors='darkred', linewidths=0.5)
    ax_right.set_xlim(0, 2.5)
    ax_right.set_xlabel('Depth (km)', fontsize=7)
    ax_right.tick_params(labelsize=6)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    plt.setp(ax_right.get_xticklabels(), visible=False)

    # --- Bottom panel (lon vs depth) ---
    ax_bot.scatter(lons, depths, s=MARKER_SIZE, c=matched_times,
                   cmap=cmap, norm=norm, zorder=4,
                   edgecolors='k', linewidths=0.2, alpha=0.85)
    if core_lon is not None:
        ax_bot.scatter(core_lon, core_dep, s=50, c='red', marker='*',
                       zorder=6, edgecolors='darkred', linewidths=0.5)
    ax_bot.set_ylim(2.5, 0)   # depth increases downward
    ax_bot.set_xlabel('Longitude', fontsize=7)
    ax_bot.set_ylabel('Depth (km)', fontsize=7)
    ax_bot.xaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax_bot.tick_params(labelsize=6)

    # --- Colorbar in bottom-right cell ---
    ax_cb.set_axis_off()
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_cb, fraction=0.6, pad=0.05, aspect=15)
    cbar.set_label('Date', fontsize=7)
    n_ticks = min(5, len(matched_times))
    tick_vals = np.linspace(t_min, t_max, n_ticks)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([UTCDateTime(t).strftime('%Y-%m-%d')
                         for t in tick_vals], fontsize=5)

    fig.suptitle(f'Family {fnum}  —  {len(lats)} located events', fontsize=9)
    plt.savefig(os.path.join(detector.get('output_folder'), 'families',
                             f'map{fnum}.png'), dpi=120)
    plt.close(fig)


def load_fm_catalog(fm_mat_path):
    """
    Load the focal mechanism catalog from .mat file.

    Returns a dict mapping EventID -> struct with fields:
    avmech (strike,dip,rake), avfnorm, avslip, mechqual, faultType, lon, lat, depth.
    """
    mat = sio.loadmat(fm_mat_path, squeeze_me=True, struct_as_record=False)
    Po = mat['Po_Clu']
    return {int(e.ID): e for e in Po}


# Fault type colors (matching Figure04_FM.m convention)
_FAULT_COLORS = {'N': '#4393c3', 'R': '#d6604d', 'S': '#74c476', 'U': '#969696'}


def create_axial_fm_plot(detector, fnum, catalog_csv, fm_catalog,
                         fm_mat_path=None):
    """
    Create a cross-section focal mechanism figure for a family.

    Layout (all beachballs in lower-hemisphere projection, qual A/B only):
      Top-left  : map view — longitude (x) vs latitude (y)
      Top-right : side view — depth (x) vs latitude (y)  [shares y with map]
      Bottom-left: front view — longitude (x) vs depth (y) [shares x with map]
      Bottom-right: fault-type legend

    Parameters
    ----------
    detector : Detector object
    fnum : int
        Family number.
    catalog_csv : str
        Path to DD catalog CSV (EventID, Time, Latitude, Longitude, Depth).
    fm_catalog : dict
        Mapping of EventID -> mat_struct, from load_fm_catalog().
    fm_mat_path : str, optional
        Unused; kept for API compatibility.
    """
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec

    members = detector.get_members(fnum)
    rtimes_mpl = detector.get('rtable', 'startTimeMPL')
    event_times_sec = np.array([
        mdates.num2date(rtimes_mpl[m]).timestamp() for m in members])

    # Load DD catalog
    try:
        cat = pd.read_csv(catalog_csv, parse_dates=['Time'])
        cat['Time'] = pd.to_datetime(cat['Time'], utc=True)
        cat_times = cat['Time'].values.astype('datetime64[ns]').astype(float) / 1e9
    except Exception:
        return

    # Match each family event → DD catalog → FM catalog
    fms = []
    for t_sec in event_times_sec:
        diffs = np.abs(cat_times - t_sec)
        idx = np.argmin(diffs)
        if diffs[idx] >= 60:
            continue
        eid = int(cat.iloc[idx]['EventID'])
        if eid not in fm_catalog:
            continue
        ev = fm_catalog[eid]
        try:
            sdr = [int(ev.avmech[0]), int(ev.avmech[1]), int(ev.avmech[2])]
        except Exception:
            continue
        fms.append({
            'lon': float(ev.lon), 'lat': float(ev.lat),
            'depth': float(np.clip(ev.depth, 0, 2.5)),
            'sdr': sdr,
            'qual': str(ev.mechqual),
            'ftype': str(ev.faultType),
        })

    fms_ab = [f for f in fms if f['qual'] in ('A', 'B')]
    if not fms_ab:
        return

    bb_width = 20  # beachball width in display points

    def _draw_bb(ax, x, y, fm):
        """Draw one beachball at (x, y); fall back to scatter on error."""
        color = _FAULT_COLORS.get(fm['ftype'], '#969696')
        try:
            b = beach(fm['sdr'], xy=(x, y), width=bb_width,
                      linewidth=0.3, facecolor=color, edgecolor='k',
                      alpha=1.0, axes=ax)
            b.set_zorder(4)
            ax.add_collection(b)
        except Exception:
            ax.scatter(x, y, s=12, color=color, zorder=4,
                       edgecolors='k', linewidths=0.3)

    # --- Figure layout (same proportions as location map) ---
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[3, 1], height_ratios=[3, 1],
                  hspace=0.04, wspace=0.04)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_map)
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_map)
    ax_leg = fig.add_subplot(gs[1, 1])

    # --- Map axis setup ---
    ax_map.set_xlim(-130.05, -129.94)
    ax_map.set_ylim(45.91, 46.00)
    ax_map.xaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax_map.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax_map.tick_params(labelsize=6)
    ax_map.set_ylabel('Latitude', fontsize=7)
    plt.setp(ax_map.get_xticklabels(), visible=False)
    ax_map.plot(_CALDERA_RIM_LON, _CALDERA_RIM_LAT,
                'k-', linewidth=0.8, zorder=2)
    for name, (slon, slat) in _STATIONS.items():
        ax_map.scatter(slon, slat, marker='^', s=30, color='k',
                       facecolors='none', linewidths=0.8, zorder=5)
        ax_map.annotate(name, (slon, slat), fontsize=4,
                        xytext=(2, 2), textcoords='offset points')

    # --- Right axis setup (depth vs lat) ---
    ax_right.set_xlim(0, 2.5)
    ax_right.set_xlabel('Depth (km)', fontsize=7)
    ax_right.tick_params(labelsize=6)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    plt.setp(ax_right.get_xticklabels(), visible=False)

    # --- Bottom axis setup (lon vs depth, depth inverted) ---
    ax_bot.set_ylim(2.5, 0)
    ax_bot.set_xlabel('Longitude', fontsize=7)
    ax_bot.set_ylabel('Depth (km)', fontsize=7)
    ax_bot.xaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax_bot.tick_params(labelsize=6)

    MARKER_SIZE = 8  # small dot size for cross-section panels

    # --- Draw beachballs on map; colored dots on cross-section panels ---
    for fm in fms_ab:
        color = _FAULT_COLORS.get(fm['ftype'], '#969696')
        _draw_bb(ax_map, fm['lon'], fm['lat'], fm)
        ax_right.scatter(fm['depth'], fm['lat'], s=MARKER_SIZE, color=color,
                         zorder=4, edgecolors='k', linewidths=0.2, alpha=0.85)
        ax_bot.scatter(fm['lon'], fm['depth'], s=MARKER_SIZE, color=color,
                       zorder=4, edgecolors='k', linewidths=0.2, alpha=0.85)

    # --- Legend in bottom-right cell ---
    ax_leg.set_axis_off()
    label_map = {'N': 'Normal', 'R': 'Reverse',
                 'S': 'Strike-slip', 'U': 'Undefined'}
    handles = [
        plt.scatter([], [], s=60, color=c, edgecolors='k',
                    linewidths=0.5, label=label_map[ft])
        for ft, c in _FAULT_COLORS.items()
    ]
    ax_leg.legend(handles=handles, loc='center', fontsize=7,
                  framealpha=0.8, title='Qual A/B only', title_fontsize=7)

    fig.suptitle(
        f'Family {fnum} — Focal Mechanisms  '
        f'({len(fms_ab)} qual A/B  /  {len(fms)} total matched)',
        fontsize=9)
    plt.savefig(os.path.join(detector.get('output_folder'), 'families',
                             f'fm{fnum}.png'), dpi=130)
    plt.close(fig)


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
