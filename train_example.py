import malan
from malan.reader import Reader
import tensorflow as tf
import numpy as np
import os, time, sys, traceback

def sample_func(sample, *args):
    features = []
    labels = []
    for i,element in enumerate(sample):
        if i==0:
            labels.append(float(element))
        if isinstance(element, str):
            ft = malan.reader.string2int(element)
            features.append(ft)
        elif isinstance(element, int):
            features.append(element)
        else:
            pass
    return labels, features

def build_graph(label, features):

    return None

def start_training():
    file_path = '/Users/fan/Malanshan/storage/dataset/train/part_1'
    filenames = []
    for r,d,f in os.walk(file_path):
        for x in f:
            a_file = None
            if 'context' in x:
                a_file = r+'/'+x
                filenames.append(a_file)
    print(filenames)
    rd = Reader(filenames)
    dataset = rd.dataset(tensor_types=(tf.float32, tf.int32),
                         sample_deal_func = sample_func, generator_limit=10)
    print('check0, ', dataset)

    iterator = dataset.make_initializable_iterator()
    print('check1, ', iterator)
    init = iterator.initializer
    next_batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(13):
            res = sess.run(next_batch)
            print(res)

if __name__ == "__main__":
    start_training()