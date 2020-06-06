"""
Deal with the discrete terms in the data.

1. Map user id (did) to list of vid.
2. Map vid to integer.
"""

import numpy as np
import json
import pickle
from subprocess import getstatusoutput
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
        full_uid_vid_map = _get_full_user_map_parallel(path, num_parallel_reads)

    else:
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

    channel = Queue(maxsize=para)

    def functor(idx, q, sub_filenames):
        collector = dict()
        for i, a_file in enumerate(filenames):
            df = reader.load_data(a_file)
            uids = df.did.values
            watches = df.watch.values
            uid_vid_map = get_user_watch_map(uids, watches)
            full_uid_vid_map.update(uid_vid_map)
            collector.update(full_uid_vid_map)
        status, output = getstatusoutput('free -g')
        print('done with sub_files: {}, mem:\n{}'.format(sub_filenames, output))
        q.put(collector, block=True)
        status, output = getstatusoutput('free -g')
        print('put into queue, mem: {}'.format(output))
    procs = [Process(target=functor, args=(i, channel, filenames[i::para])) for i in range(para)]
    for x in procs:
        x.start()

    full_collector = dict()
    for i in range(para):
        sub_map = channel.get(block=True)
        full_collector.update(sub_map)
        status, output = getstatusoutput('free -g')
        print('collect {}th map, mem: {}'.format(i, output))
    return full_collector

def precache_user_map(path, cache_file, num_parallel_precache):
    full_user_map = get_full_user_map(path, num_parallel_reads=num_parallel_precache)
    with open(cache_file, 'wb') as f_save:
        pickle.dump(full_user_map, f_save)

def load_user_map_from_cache(cache_file):
    assert os.path.exists(cache_file), 'cache_file: {} do not exists'.format(cache_file)
    with open(cache_file, 'rb') as f:
        full_user_map = pickle.load(f)

    return full_user_map

def precache_context(source_path, target_path, num_parallel_precache):
    """
    Cache a context to: label, [vid1, ..., vidn, vid_chosen] format. vids are chosen by timestamp in user watch info.
    :param source_path: A string, indicates the path pointed to the directory store the parquet files.
    :param target_path: A string, indicates the path that save the
            cache files. if the dir not exist, create one (recursively).
    :param num_parallel_precache: int. number of caching concurrently.
            It will not exceed the number of context files.
    :return: List. Two lists of names of the cached files.
            [labels.pickle, ... , labels.pickle], [vids.pickle, ... , vids.pickle].
    """
    # Prepare the directory.
    #


