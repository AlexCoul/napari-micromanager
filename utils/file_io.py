#!/usr/bin/python
'''
----------------------------------------------------------------------------------------
File Input / Output utils
----------------------------------------------------------------------------------------
Alexis Coullomb
2022-09-02
alexis.coullomb.pro@gmail.com
----------------------------------------------------------------------------------------
'''

import pandas as pd
import warnings
import json

def load_positionlist(path_position, round=2):
    """
    Parse a positionList file saved from micromanager or napari-mm.

    Parameters
    ----------
    path_position : str
        Path to the positionList file.
    round : int
        If not None, round the values.

    Returns
    -------
    pos_df : dataframe
        Dataframe with x, y, z columns, and one position per row.

    """

    if str(path_position).endswith('.csv'):
        # file created by napari-mm, with x/y/z data only
        pos_df = pd.read_csv(path_position)
        
    elif str(path_position).endswith('.pos'):
        # file created by micromanager, with complex json structure
        with open(path_position, 'r') as f:
            pos = json.load(f)
        parsed_pos = {}
        # iterate over positions
        for pos_id, pos_data in enumerate(pos['map']['StagePositions']['array']):
            # make dictionnary of current position
            curr_pos = {}
            # iterate over z and xy stage devices
            for pos_axis in pos_data['DevicePositions']['array']:
                label = pos_axis['Device']['scalar']
                value = pos_axis['Position_um']['array']
                if 'x' in label.lower() and len(value) == 2:
                    curr_pos['x'] = value[0]
                    curr_pos['y'] = value[1]
                elif 'z' in label.lower() and len(value) == 1:
                    curr_pos['z'] = value[0]
                else:
                    warnings.warn("Couldn't find a device with 'xy' with 2 values,\
                    nor 'z' with 1 value, please check the parsing of\
                    the positionList file.", RuntimeWarning)
                parsed_pos[pos_id] = curr_pos
        pos_df = pd.DataFrame(parsed_pos).T
    # elif str(path_position).endswith('.json'):
    #     # file created by napari-mm, with x/y/z data only
    #     with open(path_position, 'r') as f:
    #         pos = json.load(f)
    #     pos_df = pd.DataFrame(pos).T

    if round is not None:
        pos_df = pos_df.round(decimals=round)

    return pos_df