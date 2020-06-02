import malan
from malan.reader import Reader
import tensorflow as tf
import numpy as np
import os, time, sys, traceback
import rest

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


class Trainer(object):
    def __int__(self):
        self.nn_size = [1024, 512, 256, 1]
        self.dense_weight = []

    def build_graph(self, io):
        label, features = io
        self.ctx_var = rest.SparseVariable(
            features=features,
            dim=1,
            devices="/CPU:0",
            limit=1000000,
            limit_mode='timestamp',
            check_interval=1000,
            name="ctx_var",
        )
        self.logits = tf.reduce_sum(self.ctx_var.op, axis=1)
        predict = tf.nn.sigmoid(self.logits)
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=label)
        loss = tf.reduce_mean(entropy, name='loss')
        sparse_opt = rest.SparseAdagradOptimizer(0.1, initial_accumulator_value=0.000001)
        dense_opt = tf.train.AdagradOptimizer(0.01, initial_accumulator_value=0.0000001)
        update = tf.group([
                           sparse_opt.minimize(loss, var_list=[self.ctx_var]),
                           dense_opt.minimize(loss, var_list=self.dense_weights)])


    def fc(self, inputs, layer, w_shape, b_shape, name):
        with tf.device('/CPU:0'):
            weight_initializer = tf.random_normal(w_shape, mean=0.0, stddev=self.nn_stddev[layer])
            weight = tf.get_variable(name = "%s_weights" % name,
                                     initializer = weight_initializer,
                                     trainable = True)
            bias_initializer = tf.random_normal(b_shape, mean=0.0, stddev=self.nn_bias_stddev[layer])
            bias = tf.get_variable(name = "%s_bias" % name,
                                   initializer = bias_initializer,
                                   trainable = True)
        self.dense_weights.append(weight)
        self.dense_weights.append(bias)
        return tf.nn.xw_plus_b(inputs, weight, bias)

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

    iterator = dataset.make_initializable_iterator()
    init = iterator.initializer
    next_batch = iterator.get_next()

    trainer = Trainer()
    labels, predict, loss = trainer.build_graph(next_batch)

    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        sess.run(init)
        for i in range(10):
            res = sess.run(predict)
            print(res)

if __name__ == "__main__":
    start_training()