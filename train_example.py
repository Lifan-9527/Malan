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
    def __init__(self, config):
        self.config = config
        self.nn_size = [1024, 512, 256, 1]
        self.dense_weights = []
        self.batch_size = config['batch_size']
        self.embedding_size = config['emb_size']
        self.feature_num = config['feature_num']

    def build_graph(self, io):
        label, features = io
        label = tf.reshape(label, [self.batch_size, 1])
        print(label, features)
        #features = tf.reshape(features, (self.batch_size, -1))
        
        features = tf.reshape(features, [self.batch_size * self.feature_num, 1])
        self.ctx_var = rest.SparseVariable(
            features=features,
            dim=self.embedding_size,
            devices="/CPU:0",
            check_interval=1000,
            name="ctx_var",
        )

        sparse_embedding = tf.reshape(self.ctx_var.op, [-1, 1])

        indices = [0] * self.embedding_size * self.batch_size * self.feature_num
        for i in range(len(indices)):
            indices[i] = int(i / (self.embedding_size *  self.feature_num))
        print('indices', indices, len(indices))

        deep = tf.math.segment_sum(sparse_embedding, indices)
        deep = tf.reshape(deep, (self.batch_size, 1))

        #####
        self.logits = deep
        predict = tf.nn.sigmoid(self.logits)

        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=label)
        loss = tf.reduce_mean(entropy, name='loss')
        sparse_opt = rest.SparseAdagradOptimizer(0.1, initial_accumulator_value=0.000001)
        dense_opt = tf.train.AdagradOptimizer(0.01, initial_accumulator_value=0.0000001)
        #update = tf.group([
        #                   sparse_opt.minimize(loss, var_list=[self.ctx_var]),
        #                   dense_opt.minimize(loss, var_list=self.dense_weights)])
        update = sparse_opt.minimize(loss, var_list=[self.ctx_var])
        #update = None

        return label, predict, loss, update


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

def start_training(config):
    file_path = '/dockerdata/oppen/playground/ft_local/Malanshan/storage/dataset/train/part_1'
    filenames = []
    for r,d,f in os.walk(file_path):
        for x in f:
            a_file = None
            if 'context' in x:
                a_file = r+'/'+x
                filenames.append(a_file)
    print(filenames)
    rd = Reader(filenames, config)
    dataset = rd.dataset(tensor_types=(tf.float32, tf.int32),
                         sample_deal_func = sample_func, generator_limit=None,
                         batch_size = config['batch_size'])

    iterator = dataset.make_initializable_iterator()
    init = iterator.initializer
    next_batch = iterator.get_next()

    trainer = Trainer(config)
    labels, predict, loss, update = trainer.build_graph(next_batch)
    with tf.Session() as sess:
        sess.run(init)
        
        for step in range(1000):
            sess.run(update)

            if step % config['test_leap'] == 0:
                res = sess.run(loss)
                print('step: {}, loss: {}'.format(step, res))



if __name__ == "__main__":
    config = {
        'batch_size': 32,
        'test_leap': 100,
        'test_epochs': 32,
        'save_leap': 2000,
        'emb_size': 8,
        'feature_num': 11,
    }
    try:
        start_training(config)
    except:
        print(traceback.format_exc())
