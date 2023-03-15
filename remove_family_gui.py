# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Run this script to manually remove families/clusters (e.g., correlated noise that made it
past the 'junk' detector) using a GUI interface. Reclusters and remakes images when done.
Note: using large NCOLS may make the window too wide for your monitor, and the GUI does
not currently support side scrolling...

usage: removeFamilyGUI.py [-h] [-v] [-c CONFIGFILE]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         increase written print statements
  -c CONFIGFILE, --configfile CONFIGFILE
                        use configuration file named CONFIGFILE instead of
                        default settings.cfg
  -n NCOLS, --ncols NCOLS
                        adjust number of columns in layout (default 3)
  -m MINCLUST, --minclust MINCLUST
                        only look at clusters with numbers at or above MINCLUST
"""
import argparse
import glob
import os
import tkinter as tk
from PIL import Image

import numpy as np

import redpy


class ChooserGUI:

    def __init__(self, master, configfile='settings.cfg', ncols=3, minclust=0,
                 verbose=False):
        """
        Initialize the window.

        Parameters
        ----------
        things
        """
        self.master = master
        self.minclust = minclust

        self.master.title('Opening file... please wait...')
        # Open stuff
        self.h5file, self.rtable, self.otable, self.ttable, self.ctable, _, \
        self.dtable, self.ftable, self.opt = \
            redpy.table.open_with_cfg(configfile, verbose)
        # Make images
        create_gifs(self.ftable, minclust, ncols, self.opt)
        members = [len(np.fromstring(
            self.ftable[fnum]['members'], dtype=int, sep=' ')
            ) for fnum in range(self.ftable.attrs.nClust)]

        self.master.title(f'REDPy - {self.opt.groupName} - Check '
                          'families to permanently remove')
        self.canvas = tk.Canvas(master, borderwidth=0, width=560*ncols,
                                height=1500, background='#ececec')
        self.frame = tk.Frame(self.canvas, background='#ececec')
        self.vsb = tk.Scrollbar(master, orient='vertical',
                                command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.create_window((4,4), window=self.frame, anchor='nw')
        self.frame.bind('<Configure>', lambda event,
                        canvas=self.canvas: canvas.configure(
                            scrollregion=canvas.bbox('all')))

        # Build grid of families
        r = 1
        c = 1
        self.imgobj = []
        self.invimgobj = []
        self.check = []
        self.var = []
        for n in range(minclust, self.ftable.attrs.nClust):
            self.imgobj.append(tk.PhotoImage(file=os.path.join(
                self.opt.output_folder,'clusters',f'{n}.gif')))
            self.invimgobj.append(tk.PhotoImage(file=os.path.join(
                self.opt.output_folder,'clusters',f'{n}_inv.gif')))
            self.var.append(tk.IntVar())
            self.check.append(tk.Checkbutton(
                self.frame, text=f'Family {n}: {members[n]} Members',
                image=self.imgobj[n-minclust], compound='top',
                variable = self.var[n-minclust],
                selectimage=self.invimgobj[n-minclust]).grid(
                    column=c, row=r, sticky='N'))
            c = c+1
            if c == ncols+1:
                c = 1
                r = r+1
                if r > 255:
                    print('Ran out of rows. Use -n or -m flags '
                          'to view more...')

        # Add buttons
        tk.Button(self.frame, text='Remove Checked', background='#ececec',
                  command=self.remove).grid(column=1, row=r+1,
                                            columnspan=ncols, sticky='N')
        tk.Button(self.frame, text='Cancel', background='#ececec',
                  command=self.close).grid(column=1, row=r+2,
                                           columnspan=ncols, sticky='S')

        # Bind MouseWheel, Return, Escape keys to be more useful
        self.master.bind_all('<MouseWheel>', self.mouse_wheel)
        self.master.bind('<Return>', self.remove)
        self.master.bind('<Escape>', self.close)

        # Add some padding
        for child in self.frame.winfo_children():
            child.grid_configure(padx=15, pady=15)

    def mouse_wheel(self, event):
        self.canvas.yview_scroll(-1*(event.delta), 'units')

    def remove(self, *args):
        fam_list = []
        for n in range(len(self.var)):
            if self.var[n].get() > 0:
                fam_list.append(n+self.minclust)
        if len(fam_list) > 0:
            print('\n You have selected the following families to remove:')
            print(f'{fam_list}\n')
            self.master.title('Removing... please wait...')
            redpy.table.remove_families(
                self.rtable, self.ctable, self.dtable, self.ftable, fam_list,
                self.opt)
            redpy.plotting.generate_all_outputs(
                self.rtable, self.ftable, self.ttable, self.ctable,
                self.otable, self.opt)
        else:
            print('\nNo families selected.\n')
        self.h5file.close()
        self.master.destroy()

    def close(self, *args):
        print('\nNo families selected.\n')
        self.h5file.close()
        self.master.destroy()


def parse():
    """
    Define and parse acceptable command line inputs.

    Returns
    -------
    args : ArgumentParser object

    """
    parser = argparse.ArgumentParser(description=
        'Run this script to manually remove families/clusters using a GUI')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
        help='increase written print statements')
    parser.add_argument('-c', '--configfile', default='settings.cfg',
        help=('use configuration file named CONFIGFILE instead of default settings.cfg'))
    parser.add_argument('-n', '--ncols', default=3, type=int,
        help='adjust number of columns in layout (default 3)')
    parser.add_argument('-m', '--minclust', default=0, type=int,
        help='only look at clusters with numbers at or above MINCLUST')
    args = parser.parse_args()
    return args


def create_gifs(ftable, minclust, ncols, opt):
    for n in range(minclust, min(ftable.attrs.nClust, 255*ncols-minclust)):
        im = Image.open(os.path.join(
            opt.output_folder, 'clusters', f'{n}.png')).convert('RGB')
        im.save(os.path.join(opt.output_folder,'clusters',f'{n}.gif'))
        # Create 'inverted' selection image
        source = im.split()
        blk = source[1].point(lambda i: i*0)
        source[1].paste(blk)
        source[2].paste(blk)
        invim = Image.merge('RGB', source)
        invim.save(os.path.join(opt.output_folder,'clusters',f'{n}_inv.gif'))


def delete_gifs(opt):
    if opt.verbose:
        print('Cleaning up .gif files...')
    gif_list = glob.glob(os.path.join(opt.output_folder,'clusters','*.gif'))
    for gif in gif_list:
        os.remove(gif)


def main():
    args = parse()
    remove_family_gui(**vars(args))
    print('Done')


def remove_family_gui(configfile='settings.cfg', ncols=3, minclust=0,
                      verbose=False):
    """
    """
    root = tk.Tk()
    gui = ChooserGUI(root, configfile, ncols, minclust, verbose)
    root.mainloop()


if __name__ == '__main__':
    main()


#     args = parse()
#     configfile = args.configfile
#     verbose = args.verbose
#     minclust = args.minclust
#     ncols = args.ncols
#
#
#     h5file, rtable, otable, ttable, ctable, _, dtable, ftable, opt = \
#         redpy.table.open_with_cfg(configfile, verbose)
#
#
#     # Create GUI window
#     root = tk.Tk()
#     root.title('REDPy - Check Families to Permanently Remove')
#     canvas = tk.Canvas(root, borderwidth=0, width=560*ncols, height=1500,
#                        background='#ececec')
#     frame = tk.Frame(canvas, background='#ececec')
#     vsb = tk.Scrollbar(root, orient='vertical', command=canvas.yview)
#     canvas.configure(yscrollcommand=vsb.set)
#     vsb.pack(side='right', fill='y')
#     canvas.pack(side='left', fill='both', expand=True)
#     canvas.create_window((4,4), window=frame, anchor='nw')
#     frame.bind('<Configure>', lambda event,
#                canvas=canvas: onFrameConfigure(canvas))
#
#     # Create images
#     create_gifs(ftable, minclust, ncols, opt)
#
#     # Build grid of families
#     r = 1
#     c = 1
#     imgobj = []
#     invimgobj = []
#     check = []
#     var = []
#     for n in range(minclust, ftable.attrs.nClust):
#         imgobj.append(tk.PhotoImage(file=os.path.join(
#             opt.output_folder,'clusters',f'{n}.gif')))
#         invimgobj.append(tk.PhotoImage(file=os.path.join(
#             opt.output_folder,'clusters',f'{n}_inv.gif')))
#         var.append(tk.IntVar())
#         check.append(tk.Checkbutton(frame, text=n, image=imgobj[n-minclust],
#                                     compound='top', variable = var[n-minclust],
#                                     selectimage=invimgobj[n-minclust]).grid(
#                                         column=c, row=r, sticky='N'))
#         c = c+1
#         if c == ncols+1:
#             c = 1
#             r = r+1
#             if r > 255:
#                 print("Ran out of rows. Use -n or -m flags to view more...")
#
#     # Add buttons
#     tk.Button(frame, text='Remove Checked', background='#ececec', command=remove).grid(
#         column=1, row=r+1, columnspan=ncols, sticky='N')
#     tk.Button(frame, text='Cancel', background='#ececec', command=close).grid(
#         column=1, row=r+2, columnspan=ncols, sticky='S')
#
#     # Bind MouseWheel, Return, Escape keys to be more useful
#     root.bind_all('<MouseWheel>', mouse_wheel)
#     root.bind('<Return>', remove)
#     root.bind('<Escape>', close)
#
#     # Add some padding
#     for child in frame.winfo_children(): child.grid_configure(padx=15, pady=15)
#
#     # Go!
#     root.mainloop()
#
#
#     delete_gifs(opt)
#     redpy.plotting.remove_old_files(ftable, opt)
#
#     h5file.close()
