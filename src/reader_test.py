import reader
import tensorflow as tf
import numpy as np
import os, time, sys, traceback

def reader_test():
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
    gen = rd.sample_generator()
    for i in range(100000):
        sample = next(gen)
        #print(sample)
        #if i % 1000 == 0:
        print(sample)

if __name__ == "__main__":
    scale = time.time()
    try:
        reader_test()
    except Exception as ex:
        print(ex)
        print(type(ex))
        print(isinstance(ex, StopIteration))
        #print(traceback.format_exc())
        print(time.time() - scale)
