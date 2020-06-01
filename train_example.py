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
            labels.append(element)
        if isinstance(element, str):
            ft = malan.reader.string2int(element)
            features.append(ft)
        elif isinstance(element, int):
            features.append(element)
        else:
            pass
    return [labels, features]

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
gen = rd.sample_generator(gen_func=sample_func, limit=10)
for i in range(100000):
    sample = next(gen)
    print(sample)

