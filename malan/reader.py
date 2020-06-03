# Module import
import os, time, argparse, sys, gzip, mmap, struct
import tensorflow as tf
import sklearn
import pyarrow as pa
import numpy as np
import pandas as pd
import traceback
import subprocess
import pandas as pd
import pyarrow.parquet as pq
import hashlib

import multiprocessing
from multiprocessing import Queue, Process, Value

# Submodule import
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.client import timeline

# Third party import

def load_data(file_path):
    """
    """
    table = pq.read_table(file_path)
    df = table.to_pandas()
    return df

def string2int(str_feature):
    assert isinstance(str_feature, str), "invalid string2int input"
    encode_feature = str.encode(str_feature)
    tmp = hashlib.blake2b(encode_feature, digest_size=6).digest()
    return int.from_bytes(tmp, byteorder='little')

class Reader(object):
    """
    Reader wrapped all input related functions.
    Source data is in format Apache Parquet, a pandas style format.
    A collective group communication scheme for high performance IO is under consideration.
    """
    def __init__(self, filenames, batch_size=1, worker_index=0):
        self.filenames = filenames
        self.batch_size = batch_size
        self.worker_index = worker_index

    def sample_generator(self):
        """
        sample_generator construct a generator for producing one sample each time.
        :param limit: Number of samples allowed to throw out.
        :yield features: One sample.
        """
        gen_func = self.sample_func
        assert gen_func != None, "Must use a gen_func to parse the data"
        for idx, a_file in enumerate(self.filenames):
            print('ready to deal with file: {}, schedule: {}'.format(a_file, idx/len(self.filenames)))
            df = load_data(a_file)
            datas = df.values
            idx = 0
            num_samples = datas.shape[0]
            while idx < num_samples:
                features = gen_func(datas[idx])
                if features == None:
                    idx += 1
                    continue

                # yield a list of lists
                yield features
                idx += 1
                if self.generator_limit==None:
                    continue
                else:
                    if idx>=self.generator_limit:
                        break

    def _get_batch_detached(self, num_parallel_reads=1, max_qsize=1):
        """ TODO
        Assemble batches, multiprocessing based.
        :param num_parallel_reads: number of processes concurrently running to get batches
        """
        assert isinstance(num_parallel_reads, int), "num_parallel_reads must be an integer."
        assert num_parallel_reads <= len(self.filenames), "num_parallel_reads must not exceed file numbers."

        self.q = [Queue(maxsize=max_qsize) for _ in range(num_parallel_reads)]

        shard_filenames = []
        for i in range(num_parallel_reads):
            shard_filenames += self.filenames[i::num_parallel_reads]

        def _get_sample(idx, filenames, q):
            gen = self.sample_generator(filenames)
            batch = []
            while True:
                try:
                    sample = next(gen)
                    batch.append(sample)
                    q.put(batch, block=True)
                except Exception as ex:
                    if isinstance(ex, StopIteration):
                        print('[notice] sub-reader :{} is finished.'.format(idx))
                    else:
                        print('[error] {}'.format(ex))
                        print(traceback.format_exc())

    def dataset(self, num_parallel_reads=1, tensor_types=None,
                sample_deal_func=None, generator_limit=None, batch_size=1,
                prefetch_depth=1, cache='', repeat_times=0, shuffle_buffer_size=0):
        self.sample_func = sample_deal_func
        self.generator_limit = generator_limit
        if num_parallel_reads==1:
            ds = tf.data.Dataset.from_generator(self.sample_generator, tensor_types)
            if batch_size > 1 and isinstance(batch_size, int):
                ds = ds.batch(batch_size)
            if prefetch_depth > 1 and isinstance(prefetch_depth, int):
                ds = ds.prefetch(prefetch_depth)
            if len(cache) > 0 and isinstance(cache, str):
                ds = ds.cache(cache)
            if repeat_times > 0 and isinstance(repeat_times, int):
                ds = ds.repeat(count=repeat_times)
            if shuffle_buffer_size > 0:
                ds = ds.shuffle(shuffle_buffer_size)

            return ds

        if num_parallel_reads > 1:
            ds = None

        return ds