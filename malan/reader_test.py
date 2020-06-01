import reader
import tensorflow as tf
import numpy as np
import os, time, sys, traceback

def sample_func(sample, *args):
    features = []
    for element in sample:
        if isinstance(element, str):
            ft = reader.string2int(element)
            features.append(ft)
        elif isinstance(element, int):
            features.append(element)
        else:
            pass
    return features

def sample_generator_test():
    file_path = '/Users/fan/Malanshan/storage/dataset/train/part_1'
    filenames = []
    for r,d,f in os.walk(file_path):
        for x in f:
            a_file = None
            if 'context' in x:
                a_file = r+'/'+x
                filenames.append(a_file)
    print(filenames)
    rd = reader.Reader(filenames)
    gen = rd.sample_generator(gen_func=sample_func, limit=10)
    for i in range(100000):
        sample = next(gen)
        print(sample)

if __name__ == "__main__":
    scale = time.time()
    try:
        sample_generator_test()

    except Exception as ex:
        print(ex)
        print(type(ex))
        print(isinstance(ex, StopIteration))
        #print(traceback.format_exc())
        print(time.time() - scale)
