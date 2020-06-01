import reader
import tensorflow as tf
import numpy as np
import os, time, sys

def reader_test():
    file_path = '/Users/fan/Malanshan/storage/dataset/train/'
    filenames = []
    for r,d,f in os.walk(file_path):
        a_file = ''
        for x in f:
            if 'context' in f:
                a_file = r+'/'+x
            filenames.append(a_file)
    print(filenames)
    rd = reader.Reader(filenames)
    gen = rd.sample_generator()
    for i in range(10):
        sample = next(gen)
        print(sample)