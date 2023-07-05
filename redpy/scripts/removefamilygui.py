# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Manually remove one or more families using a GUI.

Images are set up in a grid, and may be clicked to be chosen. Below each
image is the family number and how many members are in that family. When
clicked, a checkbox next to the image is checked and the background color
turns from white to red. Below the images are two buttons. "Remove Selected"
removes the selected families, and "Cancel" exits without removing anything.
These buttons may also be accessed by pressing <Enter> and <Esc>,
respectively. Closing the window also exits without removing any families.
Vertical scrolling with the mouse wheel is supported, but not horizontal
scrolling. The user may alter the number of columns (e.g., if they have a
wide or narrow screen), but only 250 rows of images can be rendered at once.

usage: redpy-remove-family-gui [-h] [-v] [-c CONFIGFILE] [-n NCOLS]
                               [-m MINFAM]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -n NCOLS, --ncols NCOLS
                        adjust number of columns in layout (default 3)
  -m MINFAM, --minfam MINFAM
                        only look at families with numbers at or above
                        MINFAM
"""
import argparse
import glob
import os
import tkinter as tk
from collections import defaultdict
from PIL import Image

import redpy


class RemoveFamilyGUI(tk.Tk):
    """Graphical User Interface (GUI) to remove families by image."""

    def __init__(self, configfile='settings.cfg', ncols=3, minfam=0,
                 verbose=False):
        """
        Load tables and create the GUI.

        Parameters
        ----------
        configfile : str, optional
            Name of configuration file to read.
        ncols : int, optional
            Number of columns in layout.
        minfam : int, optional
            Starting family number. May be used if there are too many
            families to render at once.
        verbose : bool, optional
            Enable additional print statements.

        """
        tk.Tk.__init__(self)
        self.detector = redpy.Detector(configfile, verbose, opened=True)
        self.params = {'ncols': ncols, 'minfam': minfam,
                       'maxfam': 250*ncols-minfam}
        self.objdict = defaultdict(list)  # Will hold images, check, and var
        self._create_gifs()
        self._build_frame()
        self._build_grid()
        self._add_buttons()
        self._pad()
        self.bind_all('<MouseWheel>', self.mouse_wheel)
        self.bind('<Return>', self.remove)
        self.bind('<Escape>', self.close)

    def close(self, *_):
        """Close window without selecting any families."""
        print('\nNo families selected.\n')
        self.exit()

    def exit(self):
        """Clean up and then exit."""
        self._delete_gifs()
        self.detector.close()
        self.destroy()

    def mouse_wheel(self, event):
        """Handle vertical scroll with mouse wheel."""
        self.canvas.yview_scroll(-1*(event.delta), 'units')

    def remove(self, *_):
        """Remove selected families then exit."""
        fam_list = []
        for fam, var in enumerate(self.objdict['var']):
            if var.get() > 0:
                fam_list.append(fam+self.params['minfam'])
        if len(fam_list) > 0:
            print('\nYou have selected the following families to remove:')
            print(' '.join([str(fam) for fam in fam_list])+'\n')
            self.detector.remove('family', fam_list)
            self.detector.output()
        else:
            print('\nNo families selected.\n')
        self.exit()

    def _add_buttons(self):
        """Add 'Remove Selected' and 'Cancel' buttons at bottom."""
        tk.Button(self.frame, text='Remove Selected', background='#ececec',
                  command=self.remove).grid(column=1, row=self.row+1,
                                            columnspan=self.params['ncols'],
                                            sticky='N')
        tk.Button(self.frame, text='Cancel', background='#ececec',
                  command=self.close).grid(column=1, row=self.row+2,
                                           columnspan=self.params['ncols'],
                                           sticky='S')

    def _build_frame(self):
        """Build container for GUI."""
        self.wm_title(f'REDPy - {self.detector.get("groupname")} - Select '
                      'families to permanently remove')
        self.canvas = tk.Canvas(self, borderwidth=0,
                                width=560*self.params['ncols'],
                                height=1500, background='#ececec')
        self.frame = tk.Frame(self.canvas, background='#ececec')
        self.vsb = tk.Scrollbar(self, orient='vertical',
                                command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.create_window((4, 4), window=self.frame, anchor='nw')
        self.frame.bind(
            '<Configure>', lambda event, canvas=self.canvas: canvas.configure(
                scrollregion=canvas.bbox('all')))

    def _build_grid(self):
        """Build grid of checkboxes with families."""
        self.row = 1
        column = 1
        number_of_members = [
            len(self.detector.get_members(fnum)) for fnum in range(
                len(self.detector.get('ftable')))]
        for fam in range(self.params['minfam'],
                         min(len(self.detector.get('ftable')),
                             self.params['maxfam'])):
            self.objdict['imgobj'].append(tk.PhotoImage(
                file=os.path.join(self.detector.get(
                    'output_folder'), 'families', f'{fam}.gif')))
            self.objdict['invimgobj'].append(
                tk.PhotoImage(file=os.path.join(self.detector.get(
                    'output_folder'), 'families', f'{fam}_inv.gif')))
            self.objdict['var'].append(tk.IntVar())
            self.objdict['check'].append(tk.Checkbutton(
                self.frame,
                text=f'Family {fam}: {number_of_members[fam]} Members',
                image=self.objdict['imgobj'][fam-self.params['minfam']],
                compound='top',
                variable=self.objdict['var'][fam-self.params['minfam']],
                selectimage=self.objdict['invimgobj'][
                    fam-self.params['minfam']]
                ).grid(column=column, row=self.row, sticky='N'))
            column += 1
            if column > self.params['ncols']:
                column = 1
                self.row += 1
        if len(self.detector.get('ftable')) > self.params['maxfam']:
            print('Ran out of rows. Use -n or -m flags to view more...')

    def _create_gifs(self):
        """Create family .gif files."""
        for fam in range(self.params['minfam'],
                         min(len(self.detector.get('ftable')),
                         250*self.params['ncols']-self.params['minfam'])):
            image = Image.open(os.path.join(self.detector.get(
                'output_folder'), 'families', f'{fam}.png')).convert('RGB')
            image.save(os.path.join(
                self.detector.get('output_folder'), 'families', f'{fam}.gif'))
            source = image.split()
            black = source[1].point(lambda i: i*0)
            source[1].paste(black)
            source[2].paste(black)
            inverse_image = Image.merge('RGB', source)
            inverse_image.save(os.path.join(self.detector.get(
                'output_folder'), 'families', f'{fam}_inv.gif'))

    def _delete_gifs(self):
        """Delete family .gif files."""
        if self.detector.get('verbose'):
            print('Cleaning up .gif files...')
        gif_list = glob.glob(os.path.join(self.detector.get('output_folder'),
                                          'families', '*.gif'))
        for gif in gif_list:
            os.remove(gif)

    def _pad(self):
        """Add padding around grid items."""
        for child in self.frame.winfo_children():
            child.grid_configure(padx=15, pady=15)


def remove_family_gui(configfile='settings.cfg', ncols=3, minfam=0,
                      verbose=False):
    """
    Run a Graphical User Interface (GUI) to remove families by image.

    Parameters
    ----------
    configfile : str, optional
        Name of configuration file to read.
    ncols : int, optional
        Number of columns in layout.
    minfam : int, optional
        Starting family number. May be used if there are too many families
        to render at once.
    verbose : bool, optional
        Enable additional print statements.

    """
    gui = RemoveFamilyGUI(configfile, ncols, minfam, verbose)
    gui.protocol("WM_DELETE_WINDOW", gui.close)
    gui.mainloop()


def main():
    """Handle run from the command line."""
    args = parse()
    remove_family_gui(**vars(args))
    print('Done')


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    ArgumentParser object

    """
    parser = argparse.ArgumentParser(
        prog='remove-family-gui',
        description='Manually remove one or more families using a GUI.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='increase written print statements')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
                        help=('use configuration file named CONFIGFILE '
                              'instead of default settings.cfg'))
    parser.add_argument('-n', '--ncols', default=3, type=int,
                        help='adjust number of columns in layout (default 3)')
    parser.add_argument('-m', '--minfam', default=0, type=int,
                        help=('only look at families with numbers at or '
                              'above MINFAM'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
