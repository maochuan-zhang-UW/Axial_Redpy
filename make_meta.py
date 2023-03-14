# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Make "meta.html" to hold multiple meta overview pages.

Run this script to generate a file "meta.html" in a specified directory and
with a list of runs. This page gathers the 'meta_recent.html' tabbed
overviews within the output directories into a single page.

usage: make_meta.py [-h] [-v] [-r RUNS] [-p PATH] [-t TOPATH]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -r RUNS, --runs RUNS  comma separated list of runs to include, matching
                        their "groupName" in their configuration files
  -p PATH, --path PATH  relative path to where "meta.html" should be
                        created; defaults to current path
  -t TOPATH, --topath TOPATH
                        relative path from "meta.html" to the runs; defaults
                        to current path
"""
import argparse
import os


def make_meta(runs='', path='.', topath='.', verbose=False):
    """
    Make "meta.html" to hold multiple meta overview pages.

    This page gathers the 'meta_recent.html' tabbed overviews within the
    output directories into a single page. This is intended to be used
    to monitor several runs simultaneously.

    Parameters
    ----------
    runs : str, optional
        Comma separated list of runs to include, matching their "groupName"
        in their configuration files.
    path : str, optional
        Relative path to where "meta.html" should be created; defaults to
        current path.
    topath : str, optional
        Relative path from "meta.html" to the runs; defaults to current
        path.
    verbose : bool, optional
        Increase written print statements.

    """
    filename = os.path.join(path, 'meta.html')
    if verbose:
        print(f'Creating {filename}...')
    if not runs:
        print('No runs supplied, assuming "default" only')
        runs = 'default'
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(r'<html><head><title>REDPy Meta Overview</title></head>')
        file.write(r'<body style="padding:0;margin:0">')
        for run in runs.split(','):
            runpath = r'/'.join([topath, run.strip(), 'meta_recent.html'])
            file.write(rf"""
                <iframe src="{runpath}" title="{run}"
                    style="height:350px;width:1300px;border:none;"></iframe>
                    """)
        file.write('</body></html>')


def main():
    """Handle run from the command line."""
    args = parse()
    make_meta(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    args : ArgumentParser Object

    """
    parser = argparse.ArgumentParser(
        description='Make "meta.html" to hold multiple meta overview pages.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-r', '--runs',
                        help=('comma separated list of runs to include, '
                              'matching their "groupName" in their '
                              'configuration files'))
    parser.add_argument('-p', '--path', default='.',
                        help=('relative path to where "meta.html" should be '
                              'created; defaults to current path'))
    parser.add_argument('-t', '--topath', default='.',
                        help=('relative path from "meta.html" to the runs; '
                              'defaults to current path'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
