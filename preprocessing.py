"""
Deal with the discrete terms in the data.

1. Map user id (did) to list of vid.
2. Map vid to integer.
"""

import numpy as np
import json
import malan
from malan import utils
from malan import reader
import  traceback

def map_user_to_vids(user_list):
    user2vids_dict = dict()
    user_list_arr = np.asarray(user_list)
    unique_user = np.unique(user_list_arr)

def parse_user_watch_info(pair_string):
    """
    :param pair_string: a string in numpy print format. "[[214125, 35525], [51515, 632626]]". [timestamp, vid]
    :return: timestamp, vid, numpy.ndarray
    """
    if len(pair_string) <= 4:
        timestamp = np.array([], dtype=np.int32)
        vid = np.array([], dtype=np.int32)
    else:
        pair_string = pair_string[2:-2]  # strip the '[' and ']'
        pairs = pair_string.split('], [')
        raw_str_data = np.char.split(pairs, sep=', ', maxsplit=2)
        #raw_str_data = raw_str_data.reshape((-1, 1))
        print(raw_str_data)
        print('-----')

        pair_data = np.asarray([np.array(x, dtype=np.uint64) for x in raw_str_data])
        print(pair_data.shape)
        print('-----')
        #pair_data = [np.fromstring(x, dtype=np.uint64) for x in pair_data]
        pair_data = np.fromstring(pair_data, dtype=np.uint64).reshape((-1, 2))

        print(pair_data, pair_data.shape)
        return pair_data



if __name__ == "__main__":
    try:

        # 1. Map user id to list of vid
        path = '/Users/fan/Malanshan/storage/dataset/train/part_5'
        print(dir(malan))
        filenames = malan.utils.path_to_list(path, key_word='user')
        print(filenames)

        for i, a_file in enumerate(filenames):
            df = malan.reader.load_data(a_file)
            #print(df)
            uid = df.did
            watch = df.watch

            # single user watch test
            #parse_user_watch_info(watch[0])

    except:
        print(traceback.format_exc())