"""
Deal with the discrete terms in the data.

1. Map user id (did) to list of vid.
2. Map vid to integer.
"""

import numpy as np
import json
#import malan
try:
    import utils
    import reader
except:
    from malan import utils
    from malan import reader

import  traceback
import time, os
from multiprocessing import Queue, Process

def map_user_to_vids(user_list):
    user2vids_dict = dict()
    user_list_arr = np.asarray(user_list)
    unique_user = np.unique(user_list_arr)

def parse_user_watch_info(pair_string):
    """
    :param pair_string: a string in numpy print format. "[[214125, 35525], [51515, 632626]]". [timestamp, vid]
    :return: pair_data: [[timestamp, vid], [...], ...]
    """
    if len(pair_string) <= 4:
        timestamp = np.array([], dtype=np.int32)
        vid = np.array([], dtype=np.int32)
    else:
        pair_string = pair_string[2:-2]  # strip the '[' and ']'
        pairs = pair_string.split('], [')
        raw_str_data = np.char.split(pairs, sep=', ', maxsplit=2)

        pair_data = np.asarray([np.array(x, dtype=np.uint64) for x in raw_str_data])
        pair_data = np.fromstring(pair_data, dtype=np.uint64).reshape((-1, 2))

        return pair_data

def get_user_watch_map(uids, watches):
    """

    :param uids: string ndarray
    :param watches: string ndarray
    :return: uid_vid_map: A dict, mapping uid to vid
    """
    assert uids.size == watches.size, "uids and watches must have same size"
    uid_vid_map = dict()

    for i, uid in enumerate(uids):
        uid_vid_map[uid] = parse_user_watch_info(watches[i])

    return uid_vid_map

def get_full_user_map(path, num_parallel_reads=None):
    assert isinstance(num_parallel_reads, int), "invalid type of num_parallel_reads."
    if num_parallel_reads > 1:
        return _get_full_user_map_parallel(path, num_parallel_reads)

    filenames = utils.path_to_list(path, key_word='user')
    full_uid_vid_map = dict()
    for i, a_file in enumerate(filenames):
        df = reader.load_data(a_file)
        uids = df.did.values
        watches = df.watch.values
        uid_vid_map = get_user_watch_map(uids, watches)
        full_uid_vid_map.update(uid_vid_map)
    return full_uid_vid_map

def _get_full_user_map_parallel(path, num_parallel_reads):
    filenames = utils.path_to_list(path, key_word='user')
    para = min(len(filenames), num_parallel_reads)

    full_uid_vid_map = dict()

    channels = [Queue(maxsize=1) for _ in range(para)]

    def functor(idx, q, sub_filenames):
        collector = dict()
        for i, a_file in enumerate(filenames):
            df = reader.load_data(a_file)
            uids = df.did.values
            watches = df.watch.values
            uid_vid_map = get_user_watch_map(uids, watches)
            full_uid_vid_map.update(uid_vid_map)
            collector.update(full_uid_vid_map)
        q.put(collector, block=True)
    procs = [Process(target=functor, args=(i, channels[i], filenames[i::para])) for i in range(para)]
    for x in procs:
        x.start()

    full_collector = dict()
    for i in range(para):
        sub_map = channels[i].get(block=True)
        full_collector.update(sub_map)
    return full_collector