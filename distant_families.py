# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Find families with distant catalog matches.

Run this script to print out the families with a minimum percentage of
regional and/or teleseismic matches contained in their .html output files
that can then be copy/pasted into remove_family.py. An optional table is
printed that summarizes matches of each type. A custom 'FINDPHRASE' may
be given to find matches in the location string

usage: distant_families.py [-h] [-v] [-c CONFIGFILE] [-f FINDPHRASE]
                           [-p PERCENT]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements, including a table
                        of matches
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -f FINDPHRASE, --findphrase FINDPHRASE
                        phrase to explicitly find in location string
  -p PERCENT, --percent PERCENT
                        minimum percentage of regional/teleseismic matches,
                        default 90
"""
import argparse
import glob
import os

import numpy as np

import redpy


class DistantCounter():
    """Counter for distant matches within family .html files."""

    def __init__(self, filename, findphrase=''):
        """
        Read file and save counts as attributes.

        Parameters
        ----------
        filename : str
            Family .html file to open.
        findphrase : str, optional
            Specific phrase to find within the location string.

        """
        self.findphrase = findphrase
        self.fnum = int(os.path.basename(filename).split('.')[0])
        with open(filename, 'r', encoding='utf-8') as file:
            data = file.read()
            self.count_regional = data.count('regional')
            self.count_tele = data.count('teleseismic')
            self.count_local = data.count('Potential local match:')
            if self.findphrase:
                self.count_findphrase = data.count(self.findphrase)
            else:
                self.count_findphrase = 0
            self.count_total = (self.count_regional + self.count_tele
                                + self.count_local)

    def get_percent(self, kind='distant'):
        """
        Get the percentage of a certain 'kind' of match out of the total.

        Parameters
        ----------
        kind : str, optional
            Phrase corresponding to the 'kind' of earthquake.
            options are:
                'distant' - Regional and teleseismic matches; default.
                'regional' - Only regional matches.
                'regional_notele' - Regional matches but teleseisms
                    excluded from the total.
                'tele' - Only teleseismic matches.
                'findphrase' - Specific phrase given at init.

        Returns
        -------
        float
            Percentage of lines that match the 'kind' specified.

        """
        if self.count_total > 0:
            if kind == 'distant':
                return 100*((self.count_regional + self.count_tele)
                            / self.count_total)
            if kind == 'regional':
                return 100*self.count_regional/self.count_total
            if kind == 'tele':
                return 100*self.count_tele/self.count_total
            if kind == 'findphrase':
                return 100*self.count_findphrase/self.count_total
            if (kind == 'regional_notele') and ((self.count_total
                                                - self.count_tele) > 0):
                return 100*self.count_regional/(self.count_total
                                                - self.count_tele)
        return 0

    def print_stats(self):
        """Print a line describing what was found."""
        if self.count_total - self.count_local + self.count_findphrase > 0:
            print_str = f'Family {self.fnum:4} : L {self.count_local:2} | '
            print_str += f'R {self.count_regional:2} | T {self.count_tele:2}'
            if self.findphrase:
                print_str += f' | F {self.count_findphrase:2}'
            print_str += f' | Distant {self.get_percent("distant"):5.1f}%'
            if self.findphrase:
                print_str += f' | "{self.findphrase}" '
                print_str += f'{self.get_percent("findphrase"):5.1f}%'
            print(print_str)

    def append_fam(self, fam_string, percent=90, kind='distant'):
        """
        Append family number to a string if it meets the criteria.

        Parameters
        ----------
        fam_string : str
            String to append family number to.
        percent : float, optional
            Percentage that must be met/exceeded to append.
        kind : str, optional
            Phrase corresponding to the 'kind' of earthquake.
            options are:
                'distant' - Regional and teleseismic matches; default.
                'regional' - Only regional matches.
                'regional_notele' - Regional matches but teleseisms
                    excluded from the total.
                'tele' - Only teleseismic matches.
                'findphrase' - Specific phrase given at init.
                'regional3' - Instead of exceeding the percentage, will
                    append if there are 3 or more regional matches.

        """
        if kind == 'regional3':
            if self.count_regional >= 3:
                fam_string += f' {self.fnum}'
        else:
            if self.get_percent(kind) >= percent:
                fam_string += f' {self.fnum}'
        return fam_string


def distant_families(configfile='settings.cfg', verbose=False, findphrase='',
                     percent=90):
    """
    Find families with distant catalog matches by parsing their .html files.

    Prints family numbers that match the criteria that can then be copied as
    arguments to other removal functions, while also allowing the user to
    vet those matches prior to removal.

    Parameters
    ----------
    configfile : str, optional
        Configuration file to open.
    verbose : bool, optional
        Enable additional print statements, including a summary table.
    findphrase : str, optional
        Specific phrase to find in location string.
    percent : float, optional
        Minimum percentage of matches required to add a family to the list;
        90% by default.

    """
    config = redpy.Config(configfile, verbose)
    fam_dict = {'distant': '',
                'regional': '',
                'regional_notele': '',
                'tele': '',
                'findphrase': '',
                'regional3': ''}
    flist = np.array(glob.glob(os.path.join(config.get('output_folder'), 'clusters',
                                            '*.html')))
    fnums = [int(os.path.basename(fname).split('.')[0]) for fname in flist]
    for fname in flist[np.argsort(fnums)]:
        counter = DistantCounter(fname, findphrase)
        if counter.count_total > 0:
            if config.get('verbose'):
                counter.print_stats()
            for key in fam_dict:
                fam_dict[key] = counter.append_fam(fam_dict[key], percent, key)
    print(f'\n{percent}%+ Teleseismic:\n{fam_dict["tele"]}\n')
    print(f'\n{percent}%+ Regional+Teleseismic:\n{fam_dict["distant"]}\n')
    print(f'\n{percent}%+ Regional:\n{fam_dict["regional"]}\n')
    print(f'\n{percent}%+ Regional (ignore Teleseisms):\n'
          f'{fam_dict["regional_notele"]}\n')
    print(f'\n3+ Regional Matches: \n{fam_dict["regional3"]}\n')
    if findphrase:
        print(f'{percent}%+ containing "{findphrase}":\n'
              f'{fam_dict["findphrase"]}\n')


def main():
    """Handle run from the command line."""
    args = parse()
    distant_families(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    args : ArgumentParser object

    """
    parser = argparse.ArgumentParser(
        description='Find families with distant catalog matches.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help=('increase written print statements, including '
                              'a table of matches'))
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-f', '--findphrase', default='',
                        help='phrase to explicitly find in location string')
    parser.add_argument('-p', '--percent', type=float, default=90,
                        help=('minimum percentage of regional/teleseismic '
                              'matches, default 90'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
