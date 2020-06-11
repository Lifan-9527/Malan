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


def precache_item_map(path, cache_file, num_parallel_precache=1):
    """
    Cache the item to a map structure.
    In result, key is vid from context. value is a int64 vector contains:
    {vid : [vid, cid, title_length, class_id, second_class, is_intact, stars]}
    with {int : numpy.ndarray 1*7}

    :param path:
    :param cache_file:
    :param num_parallel_precache:
    :return:
    """
    filenames = utils.path_to_list(path, key_word='item')
    q = Queue(maxsize=num_parallel_precache)
    para = min(num_parallel_precache, len(filenames))

    def sub_proc(sub_filenames, q, idx):
        for _, a_file in enumerate(sub_filenames):
            df = reader.load_data(a_file)

            vid = np.asarray(df.vid.values, dtype=np.int64)
            cid = np.asarray(df.cid.values, dtype=np.int64)
            title_length = np.asarray(df.title_length.values, dtype=np.int64)
            class_id = np.asarray(df.class_id.values, dtype=np.int64)
            second_class = np.asarray(df.second_class.values, dtype=np.int64)
            is_intact = np.asarray(df.is_intact.values, dtype=np.int64)
            stars = df.stars.values

            sample_member = [vid, cid, title_length, class_id, second_class, is_intact, stars]

            sub_item_map = dict()
            collector = dict()

            for i,k in enumerate(vid):
                sample = [vid[i], cid[i], title_length[i], class_id[i], second_class[i], is_intact[i]]
                sample = np.asarray(sample, dtype=np.int64)
                sample = np.concatenate([sample, stars[i]])
                #print(sample, type(sample), sample.dtype)
                sub_item_map[k] = sample
                collector.update(sub_item_map)

            q.put(collector, block=True, timeout=False)

    proc_ent = [Process(target=sub_proc, args=(filenames[_i::para], q, _i)) for _i in range(para)]
    for x in proc_ent:
        x.start()

    all_item_map = dict()
    for i in range(para):
        sub_collector = q.get(block=True, timeout=None)
        all_item_map.update(sub_collector)

    with open(cache_file, 'wb') as f_save:
        pickle.dump(all_item_map, f_save)

    return cache_file


def load_item_map(cache_file):
    """
    Load the item map.
    :param cache_file:
    :return:
    """
    with open(cache_file, 'rb') as f:
        full_item_map = pickle.load(f)
    return full_item_map


def get_vid_to_cid_map_from_item(path=None, cache_file=None, num_parallel_reads=1):
    """

    :param path:
    :param cache_file:
    :param num_parallel_reads:
    :return:
    """
    if not cache_file:
        raise NotImplementedError
    item_map = load_item_map(cache_file)
    for k, v in item_map.items():
        item_map[k] = v[1]
    vid_cid_map = item_map
    return vid_cid_map


def precache_context_to_samples(source_path, target_path, num_parallel_precache=1):
    """
    Unlike item and user info, samples from context are stored in shard files.
    According to two-tower structure
    sample are stored as:
        block1: vid, prev, mod, mf, aver, sver, region, index, vids_from_did
        block2: features_from_item_info
    block1 and block2 are stitched as a sample, while they separately fed to
    left and right tower.

    :param source_path: where the context file stored.
    :param target_path: Directory used for caching.
    :param num_parallel_precache:
    :return:
    """
    pass
