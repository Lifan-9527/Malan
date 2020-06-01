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
    tmp = hashlib.blake2b(encode_feature, digest_size=5).digest()
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
        for idx, a_file in enumerate(self.filenames):
            print('ready to deal with file: {}, schedule: {}'.format(a_file, idx/len(self.filenames)))
            df = load_data(a_file)
            datas = df.values
            idx = 0
            num_samples = datas.shape[0]
            while idx < num_samples:
                features = []
                for element in datas[idx]:
                    if isinstance(element, str):
                        ft = string2int(element)
                        features.append(ft)
                    elif isinstance(element, int):
                        features.append(element)
                    else:
                        pass
                yield features
                idx += 1
