import sklearn
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import time

import hashlib

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


def train(filename):

	x = tf.placeholder(tf.int32, shape=[None])
	#dataset = get_dataset(x)
	#iterator = tf.data.make_initializable_iterator(dataset)
	#init = iterator.initializer
	#data = iterator.get_next()

	sess = tf.Session()

	scale = time.time()
	df = load_data(filename)
	print(df.values.shape)
	print('load_data cost: {}'.format(time.time()-scale))
	scale = time.time()
	start_time, step_time = 0, 0

	#sess.run(init, feed_dict={x: [1,2,3,4,5,6,7,8,9,10,11]})

	for i, samples in enumerate(df.values):
		step_start_time = time.time()
		#if i % 100 == 0:
		#print('step_time: {}, schedule: {}'.format(step_time, i / df.values.shape[0]))

		features = []
		for t in samples:
			if isinstance(t, str):
				ft = string2int(t)
				features.append(ft)
			elif isinstance(t, int):
				features.append(t)
			else:
				pass

		#features = np.array(features, dtype=np.int32)
		#features = features.reshape([1, -1])
		print(features)
		res = sess.run(x, feed_dict={x: features})
		print(res)

		step_time = time.time() - step_start_time
	print('parse values cost: {}'.format(time.time() - scale))


if __name__ == "__main__":
	filename = "../storage/dataset/train/part_1/context.parquet"
	train(filename)
