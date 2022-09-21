""" Utils for loading events.
"""
import h5py
import numpy as np
import pandas as pd
import os.path as op
from dv import AedatFile
from math import floor, ceil
from tqdm import trange

def extract_aedat4(path):
    """ Extract events from AEDAT4 file.
        Args:
            path: str, the path of input aedat4 file.
        Returns:
            events: pd.DataFrame, pandas data frame containing events.
    """
    with AedatFile(path) as f:
        events = np.hstack([packet for packet in f['events'].numpy()])
    events = pd.DataFrame(events)[['timestamp', 'x', 'y', 'polarity']]
    events = events.rename(columns={'timestamp': 't', 'polarity': 'p'})
    return events


def load_events(path, slice=None, to_df=True, start0=False, verbose=False):
    """ Load the DVS events in .h5 or .aedat4 format.
    Args:
        path: str, input file name.
        slice: tuple/list, two elements, event stream slice start and end.
        to_df: whether turn the event stream into a pandas dataframe and return.
        start0: set the first event's timestamp to 0.
    """
    ext = op.splitext(path)[1]
    assert ext in ['.h5', '.aedat4']
    if ext == '.h5':
        f_in = h5py.File(path, 'r')
        events = f_in.get('events')[:]
    else:
        events = extract_aedat4(path)
        events = events.to_numpy()  # .astype(np.uint32)

    if verbose:
        print(events.shape)
    if slice is not None:
        events = events[slice[0]:slice[1]]
    if start0:
        events[:, 0] -= events[0, 0]  # Set the first event timestamp to 0
        # events[:,2] = 260-events[:,2] # Y originally is upside down
    if to_df:
        events = pd.DataFrame(events, columns=['t', 'x', 'y', 'p'])
    return events
