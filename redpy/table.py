# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
Module for handling REDPy Table() object.

The Table object holds references to data stored on disk of a single type
(e.g., repeaters, families), and has methods to interact with the data in
that table.
"""
import numpy as np
import matplotlib.dates as mdates
from tables import Int32Col, Float32Col, Float64Col, StringCol, ComplexCol


_NAMES = {'ctable': ('correlation', 'Correlation Matrix'),
          'dtable': ('deleted', 'Manually Deleted Events'),
          'ftable': ('families', 'Families Table'),
          'jtable': ('junk', 'Junk Catalog'),
          'otable': ('orphans', 'Orphan Catalog'),
          'rtable': ('repeaters', 'Repeater Catalog'),
          'ttable': ('triggers', 'Trigger Catalog')}


class Table():
    """
    Interface between Detector() and tables on disk.

    Attributes
    ----------
    column_names : list
        Ordered list of columns in table.
    columns_in_memory : dict
        Columns from table that are currently duplicated in memory to reduce
        file read operations.
    long_name : str
        Long name of the table (e.g., 'repeaters'); corresponds to name of
        table in file.
    name : str
        Short name of the table in the file structure (e.g., 'rtable' for
        the 'repeaters' table).
    table : PyTables Table object
        In-memory representation of this table within the hdf5 file.
    title : str
        Title of table in file.
    """

    def __init__(self, name):
        """
        Set initial attribute structure.

        Parameters
        ----------
        name : str
            Short name of the table in the file structure (e.g., 'rtable'
            for the 'repeaters' table).

        """
        self.name = name
        self.long_name, self.title = _NAMES[name]
        self.table = None
        self.column_names = []
        self.columns_in_memory = {}

    def __len__(self):
        """Define length."""
        if self.table is not None:
            return len(self.table)
        return None  # pragma: no cover

    def __repr__(self):
        """Define representation string."""
        if self.table is not None:
            return f'redpy.Table("{self.name}") -> {self.table}'
        return f'redpy.Table("{self.name}") -> not open'

    def __str__(self):
        """Define print string."""
        if self.table is not None:
            if self.name == 'ctable':
                return f'Number of {self.long_name} pairs : {len(self.table)}'
            if self.name in ['jtable', 'dtable']:
                return f'Number of {self.long_name} events : {len(self.table)}'
            return f'Number of {self.long_name:<10s}: {len(self.table)}'
        return f'{self.name} is not open'

    def append(self, row):
        """
        Append a row or multiple rows to the table.

        Parameters
        ----------
        row : structured array, dict
            Row or rows to append to the table. If a dictionary, must
            represent the contents of a single row. The only way to
            append multiple rows is to pass a structured array, such as a
            slice from another table.

        """
        if np.shape(row) == ():  # Single row as dict or structured array
            self._append_single(row)
        else:  # pragma: no cover
            for oldrow in row:
                self._append_single(oldrow)

    def forget(self, col='all'):
        """
        Forget column(s) in memory.

        Parameters
        ----------
        col : str or str list, optional
            Name(s) of columns to remove from memory. By default or if
            'all', forgets all columns.

        """
        if isinstance(col, str):
            if col == 'all':
                self.columns_in_memory = {}
            self.columns_in_memory.pop(col, None)
        else:
            for i in col:
                self.columns_in_memory.pop(i, None)

    def get(self, col=None, row=None):
        """
        Access data from the table.

        Parameters
        ----------
        col : str or int, optional
            Column name or column number. Only supports a single column at
            a time.
        row : int or int ndarray, optional
            Row index or indices to slice.

        Returns
        -------
        array_like or Table
            If both row and column are given, returns the sliced contents of
            the column. If only a single row is given, the return will have
            the same type and shape as a single cell of that column. If
            only the column is given, returns the entire column as an array.
            If only the row is given, returns a slice of the table with all
            columns as a structured array. If neither are given, returns the
            entire Table object itself.

        """
        if isinstance(col, int):  # pragma: no cover
            col = self.column_names[col]
        if (col is not None) and (row is not None):
            # Slice of a column
            if col in self.columns_in_memory:
                return self.columns_in_memory[col][row]
            return self.table[row][col]
        if (col is not None) and (row is None):
            # Entire column
            if col in self.columns_in_memory:
                return self.columns_in_memory[col]
            return self.table.col(col)
        if row is not None:
            # Row slice of table
            return self.table[row]
        # Entire object
        return self

    def initialize(self, h5file, group, config):
        """
        Populate structure of table in an empty hdf5 file.

        Parameters
        ----------
        h5file : PyTables File object
            In-memory representation of the hdf5 file on disk.
        group : PyTables Group object
            Group within the hdf5 file where the new table will be created.
        config : Config object
            Describes the run parameters.

        """
        self.table = h5file.create_table(group, self.long_name,
                                         self._table_definition(config),
                                         self.title)
        self._initialize_attrs(config)
        self._fill_column_names()
        self.table.flush()

    def move(self, table_to, row):
        """
        Move a row or rows from this table to another table.

        Parameters
        ----------
        table_to : Table object
            Table to move rows into that shares columns with this table. If
            this table has additional columns that are not shared they will
            be dropped.
        row : int or int ndarray
            Row or row indices to move.

        """
        table_to.append(self.get(row=row))
        self.remove(row)

    def open(self, h5file, config):
        """
        Open link to specified table from file.

        Parameters
        ----------
        h5file : PyTables File object
            In-memory representation of the hdf5 file on disk.
        config : Config object
            Describes the run parameters.

        """
        if h5file:
            # pylint: disable=W0123
            # I trust use of eval() here.
            self.table = eval(
                f'h5file.root.{config.get("groupname")}.{self.long_name}')
            # pylint: enable=W0123
            self._fill_column_names()
            self._compatibility_checks()

    def populate_from_table(self, table_from, config, dsta=0):
        """
        Copy entire contents from another Table into this Table.

        Parameters
        ----------
        table_from : Table object
            Table with same 'name' to copy current contents of.
        config : Configuration object
            Describes the run parameters (for this table).
        dsta : int, optional
            Difference in 'nsta' between this table and table_from (must
            be positive).

        """
        for row_from in table_from.table.iterrows():
            row = self.table.row
            for column in self.column_names:
                if column in ['windowAmp', 'windowCoeff']:
                    row[column] = np.append(row_from[column], np.zeros(dsta))
                elif column == 'FI':
                    row[column] = np.append(
                        row_from[column], np.empty(dsta)*np.nan)
                elif column == 'waveform':
                    row[column] = np.append(
                        row_from[column], np.zeros(dsta*config.get('wshape')))
                elif column == 'windowFFT':
                    row[column] = np.append(
                        row_from[column], np.zeros(dsta*config.get('winlen')))
                else:
                    row[column] = row_from[column]
            row.append()
        if self.name == 'rtable':
            self.table.attrs.ptime = table_from.table.attrs.ptime
            self.table.attrs.previd = table_from.table.attrs.previd
        elif self.name == 'ftable':
            self.table.attrs.current_max_famlen = \
                table_from.table.attrs.current_max_famlen
            self.table.attrs.allowed_max_famlen = config.get('max_famlen')
        self.table.flush()

    def remember(self, col):
        """
        Put a column from this table into memory if it isn't there already.

        Parameters
        ----------
        col : str or str list
            Name(s) of columns to put into memory (specifically, the
            attribute 'columns_in_memory'). If 'all', puts all columns.

        Raises
        ------
        KeyError
            If a provided column name does not exist.

        """
        if isinstance(col, str):
            if col == 'all':
                col = self.column_names
            else:
                col = [col]
        for column in col:
            if column not in self.column_names:
                raise KeyError(f'{column} not a column of this table')
            if column not in self.columns_in_memory:
                self.columns_in_memory.update({column: self.get(column)})

    def remove(self, row):
        """
        Remove a row or rows from the table.

        Parameters
        ----------
        row : int, int ndarray, or str
            Row number or slice to remove. The only string argument accepted
            is 'all', which removes all rows and empties the table.

        """
        if isinstance(row, str) and (row == 'all'):
            self.table.remove_rows(0)
            self.columns_in_memory = {}
        else:
            if isinstance(row, (int, np.int32, np.int64)):
                self.table.remove_row(row)
            else:
                row = np.sort(row)[::-1]
                for i in row:
                    self.table.remove_row(i)
            if len(self.table) > 0:
                for col in self.columns_in_memory:
                    self.columns_in_memory[col] = np.delete(
                        self.columns_in_memory[col], row, axis=0)
            else:
                self.columns_in_memory = {}
        self.table.flush()

    def set(self, value, col, row=None):
        """
        Update data in table.

        Parameters
        ----------
        value : array_like
            Data to be written to the table for a single column. Should
            match the type and shape of the destination.
        col : str or int
            Name or index of the column to write to. Only supports a single
            column at a time.
        row : int or int ndarray, optional
            Row or rows to write to. If an integer, this refers to a single
            row, and thus value should be a single cell. If an array, these
            are row slices, and the length of value should match.

        Raises
        ------
        ValueError
            If None is passed as the value. Other 'empty' values like 0,
            NaN, '', or [] are accepted.

        """
        if value is None:
            raise ValueError('None as value not accepted!')
        if isinstance(col, int):  # pragma: no cover
            col = self.column_names[col]
        if row is not None:
            if not isinstance(row, (int, np.int32, np.int64)):
                for i, j in enumerate(row):
                    self.table.modify_column(start=j, column=value[i],
                                             colname=col)
                    if col in self.columns_in_memory:  # pragma: no cover
                        self.columns_in_memory[col][j] = value[i]
                col = None
            else:
                if row < 0:
                    row = len(self) + row
                start = row
                stop = row+1
                step = 1
                if col in self.columns_in_memory:
                    self.columns_in_memory[col][row] = value
        elif len(value) == len(self):
            start = 0
            stop = len(self)
            step = 1
            if col in self.columns_in_memory:  # pragma: no cover
                self.columns_in_memory[col] = value
        if col:
            self.table.modify_column(start, stop, step, value, col)
        self.table.flush()

    def _append_single(self, row):
        """Append a single row and update columns in memory."""
        newrow = self.table.row
        for col in self.column_names:
            newrow[col] = row[col]
            if col in self.columns_in_memory:
                if not isinstance(row[col], np.ndarray):
                    if np.array(self.columns_in_memory[col]).size == 0:
                        self.columns_in_memory[col] = np.array(row[col])
                    else:
                        self.columns_in_memory[col] = np.append(
                            self.columns_in_memory[col], row[col])
                else:
                    if np.array(self.columns_in_memory[col]).size == 0:
                        self.columns_in_memory[col] = np.array([row[col]])
                    else:
                        self.columns_in_memory[col] = np.append(
                            self.columns_in_memory[col], [row[col]], axis=0)
        newrow.append()
        self.table.flush()

    def _compatibility_checks(self):
        """Run compatibility checks for tables created with older versions."""
        self._check_attrs()
        self._check_epoch_date()
        self._check_duplicates()

    def _check_attrs(self):  # pragma: no cover
        """Populate missing attributes from older tables."""
        if (self.name == 'ftable') and (
                'allowed_max_famlen' not in self.table.attrs):
            self.table.attrs.allowed_max_famlen = 1000000
            self.table.attrs.current_max_famlen = np.max(
                [len(i) for i in self.get('members')])

    def _check_epoch_date(self):  # pragma: no cover
        """Check and fix epoch of matplotlib dates stored in table."""
        if len(self) > 0 and self.name in ['ttable', 'otable', 'rtable',
                                           'dtable', 'ftable']:
            column_name = 'startTimeMPL'
            if self.name == 'ftable':
                column_name = 'startTime'
            time_column = self.get(col=column_name)
            reference = [mdates.date2num(np.datetime64('now')),
                         mdates.date2num(np.datetime64('1900-01-01'))]
            if time_column[0] > reference[0]:
                time_column[time_column > reference[0]] += mdates.date2num(
                    np.datetime64('0000-12-31'))
                self.set(time_column, column_name)
            elif time_column[0] < reference[1]:
                time_column[time_column < reference[1]] += mdates.date2num(
                    np.datetime64('1970-01-01'))
                self.set(time_column, column_name)

    def _check_duplicates(self):
        """Check for duplicate entries in the correlation table."""
        if self.name == 'ctable':
            id1 = self.get('id1')
            id2 = self.get('id2')
            all_ids = np.vstack([id1, id2]).T.copy()
            dtypes = all_ids.dtype.descr * 2
            uniques = np.unique(all_ids.view(dtypes), return_index=True)[1]
            if len(uniques) < len(id1):  # pragma: no cover
                duplicates = np.setdiff1d(np.arange(len(id1)), uniques)
                print(f'Removing {len(duplicates):i} duplicate '
                      'correlation entries...')
                self.remove(duplicates)

    def _fill_column_names(self):
        """Fill names of columns from the table."""
        self.column_names = self.table.colnames

    def _initialize_attrs(self, config):
        """Set attributes to the table."""
        if self.name == 'rtable':
            self.table.attrs.scnl = [config.get('station'),
                                     config.get('channel'),
                                     config.get('network'),
                                     config.get('location')]
            self.table.attrs.samprate = config.get('samprate')
            self.table.attrs.winlen = config.get('winlen')
            self.table.attrs.ptrig = config.get('ptrig')
            self.table.attrs.atrig = config.get('atrig')
            self.table.attrs.fmin = config.get('fmin')
            self.table.attrs.fmax = config.get('fmax')
            self.table.attrs.previd = -1
            self.table.attrs.ptime = 0
        if self.name == 'ftable':
            self.table.attrs.allowed_max_famlen = config.get('max_famlen')
            self.table.attrs.current_max_famlen = 0

    def _table_definition(self, config):
        """Build and return definition of table as a dictionary."""
        if self.name in ['otable', 'rtable', 'dtable']:
            definition = {
                'id': Int32Col(shape=(), pos=0),
                'startTime': StringCol(itemsize=32, pos=1),
                'startTimeMPL': Float64Col(shape=(), pos=2),
                'waveform': Float32Col(
                    shape=(config.get('wshape')*config.get('nsta'), ), pos=3),
                'windowStart': Int32Col(shape=(), pos=4),
                'windowCoeff': Float64Col(shape=(config.get('nsta'), ), pos=5),
                'windowFFT': ComplexCol(
                    shape=(config.get('winlen')*config.get('nsta'), ),
                    itemsize=16, pos=6),
                'windowAmp': Float64Col(shape=(config.get('nsta'), ), pos=7),
                'FI': Float64Col(shape=(config.get('nsta'), ), pos=8)
                }
            if self.name == 'otable':
                definition['expires'] = StringCol(itemsize=32, pos=9)
        elif self.name == 'ctable':
            definition = {
                'id1': Int32Col(shape=(), pos=0),
                'id2': Int32Col(shape=(), pos=1),
                'ccc': Float64Col(shape=(), pos=2)
                }
        elif self.name == 'ftable':
            definition = {
                'members': StringCol(
                    itemsize=config.get('max_famlen'), shape=(), pos=0),
                'core': Int32Col(shape=(), pos=1),
                'startTime': Float64Col(shape=(), pos=2),
                'longevity': Float64Col(shape=(), pos=3),
                'printme': Int32Col(shape=(), pos=4),
                'lastprint': Int32Col(shape=(), pos=5)
                }
        elif self.name == 'jtable':
            definition = {
                'isjunk': Int32Col(shape=(), pos=0),
                'startTime': StringCol(itemsize=32, pos=1),
                'waveform': Float32Col(
                    shape=(config.get('wshape')*config.get('nsta'), ), pos=2),
                'windowStart': Int32Col(shape=(), pos=3)
                }
        elif self.name == 'ttable':
            definition = {
                'startTimeMPL': Float64Col(shape=(), pos=0)
                }
        else:
            raise ValueError(f'Unrecognized Table name: {self.name}')
        return definition
