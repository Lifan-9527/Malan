import malan

import tensorflow as tf
import numpy as np
import os, time, sys, traceback
import rest
import sklearn
from sklearn.metrics import roc_auc_score, accuracy_score
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.client import timeline

try:
    from malan import reader
    from malan import preprocessing
    from malan import utils
except:
    print(traceback.format_exc)

def roc_auc_score_FIXED(y_true, y_pred):
    if len(np.unique(y_true)) == 1: # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_pred))
    return roc_auc_score(y_true, y_pred)



class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.nn_size = config['nn_size']
        self.nn_stddev = config['nn_stddev']
        self.nn_bias_stddev = config['nn_bias_stddev']
        self.activation_type = config['activation']

        self.dense_weights = []
        self.batch_size = config['batch_size']
        self.embedding_size = config['emb_size']
        self.feature_num = config['feature_num']

    def build_graph(self, io):
        label, features = io
        label = tf.reshape(label, [self.batch_size, 1])
        #features = tf.reshape(features, (self.batch_size, -1))
        
        features = tf.reshape(features, [self.batch_size * self.feature_num, 1])
        features = tf.reshape(features, [-1, 1])
        self.ctx_var = rest.SparseVariable(
            features=features,
            dim=self.embedding_size,
            devices='/GPU:0',
            check_interval=1000,
            use_default=False,
            name="ctx_var",
        )

        sparse_embedding = tf.reshape(self.ctx_var.op, [self.batch_size, self.feature_num, self.embedding_size])

        sparse_embedding = tf.reduce_sum(sparse_embedding, axis=1)

        #indices = [0] * self.embedding_size * self.batch_size * self.feature_num
        #for i in range(len(indices)):
        #    indices[i] = int(i / (self.embedding_size *  self.feature_num))

        category = tf.reshape(sparse_embedding, (-1, self.embedding_size))

        print('check category: ', category.shape)

        deep = self.fc(category,
                0,
                [self.embedding_size, self.nn_size[0]],
                [self.nn_size[0]],
                'fc0',
                )
        if self.activation_type[0]:
            deep = getattr(tf.nn, self.activation_type[0])(deep)

        for i in range(1, len(self.nn_size)):
            deep = self.fc(
                deep,
                i,
                [self.nn_size[i-1], self.nn_size[i]],
                [self.nn_size[i]],
                'fc'+str(i),
            )
            if self.activation_type[i]:
                deep = getattr(tf.nn, self.activation_type[i])(deep)

        #####
        self.logits = deep

        predict = tf.nn.sigmoid(self.logits)

        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=label)
        loss = tf.reduce_mean(entropy, name='loss')
        auc, auc_op = tf.metrics.auc(label, predict, num_thresholds=500)
        sparse_opt = rest.SparseAdagradOptimizer(0.1, initial_accumulator_value=0.0000001)
        dense_opt = tf.train.AdagradOptimizer(0.1, initial_accumulator_value=0.0000001)
        update = tf.group([
                           sparse_opt.minimize(loss, var_list=[self.ctx_var]),
                           dense_opt.minimize(loss, var_list=self.dense_weights)])
        #update = sparse_opt.minimize(loss, var_list=[self.ctx_var])

        return label, predict, loss, auc_op, update


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

class CustomModule(object):
    def __init__(self, uid_vid_map, config):
        self.config = config
        self.uid_vid_map = uid_vid_map

    def sample_func(self, sample):
        """
        :param sample:
        :param args:
        :return: features
        """
        def get_vids(uid, timestamp):

            tstp_vids_pairs = self.uid_vid_map.get(uid, None)

            if tstp_vids_pairs is None: return None
            tstp = tstp_vids_pairs[:, 0]

            pos = np.where(tstp < timestamp)[0]
            #print('pos check', pos)

            if pos.size <= self.config['vid_window_size']:
                return None
            vids = tstp_vids_pairs[:, 1]

            filtered_vids = vids[pos]
            filtered_vids = filtered_vids[-self.config['vid_window_size']:]
            return filtered_vids


        label = sample.label.values[0]
        label = np.asarray(label, dtype=np.float32)
        uid = sample.did.values[0]
        target_vid = sample.vid.values[0]
        timestamp = sample.timestamp.values[0]

        vids = get_vids(uid, timestamp)
        if vids is None: return None
        vids = np.asarray(vids, dtype=np.int32)
        vids = vids.reshape((-1, 1))

        target_vid = np.asarray(target_vid, dtype=np.int32)
        target_vid = np.reshape(target_vid, (-1, 1))
        training_vids = np.concatenate([vids, target_vid], axis=0)
        #sys.exit(1)

        return label, training_vids



def start_training(config):
    file_path = config['file_path']

    sub_t = time.time()
    user_uid_vid_map = malan.preprocessing.get_full_user_map(file_path, num_parallel_reads=config['num_parallel_preprocess'])
    print('preprocessing user map cost {} sec.'.format(time.time() - sub_t))

    context_files = malan.utils.path_to_list(file_path, key_word='context')
    rd = reader.Reader(context_files, config)

    module = CustomModule(user_uid_vid_map, config)

    with tf.device('/CPU:0'):
        dataset = rd.dataset(tensor_types=(tf.float32, tf.int32),
                             sample_deal_func = module.sample_func, generator_limit=None,
                             batch_size = config['batch_size'])

        iterator = dataset.make_initializable_iterator()
        init = iterator.initializer
        next_batch = iterator.get_next()

        stage_cpu = data_flow_ops.StagingArea([tf.float32, tf.int32],
                                              capacity=2)
        copy_stage_cpu = stage_cpu.put(next_batch)

    with tf.device('/GPU:0'):
        stage_gpu = data_flow_ops.StagingArea([tf.float32, tf.int32],
                                              capacity=2)
        copy_stage_gpu = stage_gpu.put(stage_cpu.get())
        train_data = stage_gpu.get()

        trainer = Trainer(config)
        labels, predict, loss, auc, update = trainer.build_graph(train_data)

    init_stream = copy_stage_cpu
    stream = tf.group([copy_stage_cpu, copy_stage_gpu])

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.gpu_options.allow_growth = True
    sess_config.log_device_placement = False
    sess_config.intra_op_parallelism_threads = 40
    sess_config.inter_op_parallelism_threads = 20


    if config['timeline']:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None

    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        sess.run(init_stream)
        sess.run(stream)

        start_time, step_time, step_start_time, duration = time.time(), 0, 0, 0

        for step in range(10000):
            step_time = 0
            step_start_time = time.time()
            sess.run([stream, update], run_options=run_options, run_metadata=run_metadata)
            #_lb, _ft = sess.run(next_batch)
            #print('check next_batch = ', _lb.shape, _ft.shape)
            step_time += time.time() - step_start_time
            duration = time.time() - start_time


            if step % config['benchmark_interval']:
                print('[benchmark] step = {}, step_time = {}, duration = {}'.format(step, step_time, duration))

            # check loss, auc, and parameter size.
            if step % config['test_interval'] == 0:
                loss_metric, auc_metric = 0, 0
                feature_num = sess.run(trainer.ctx_var.size())
                for _ in range(config['test_step']):
                    _, _lb, _pred, _auc, _loss = sess.run([stream, labels, predict, auc, loss])
                    #print('check: ', _auc, _loss)

                    #auc_metric += roc_auc_score_FIXED(np.asarray(_lb), np.asarray(_pred))
                    auc_metric += _auc
                    loss_metric += _loss
                    #loss_metric += sklearn.metrics.log_loss(np.asarray(_lb), np.asarray(_pred))
                auc_metric = auc_metric / config['test_step']
                loss_metric = loss_metric / config['test_step']

                print('[in-training test] step = {}, auc = {}, loss = {}, feature_num = {}'.format(step, auc_metric, loss_metric, feature_num))

            if config['timeline']:
                if step in config['timeline_step']:
                    if step in config['timeline_steps']:
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        with open('./timeline/timeline-step{}-{}.json'.format(step, time.time()), 'w') as f:
                            f.write(ctf)
                    elif step > max(config['timeline_steps']):
                        run_options = None
                        run_metadata = None


if __name__ == "__main__":

    history_vid_num = 20,

    config = {
        'file_path': './storage/dataset/train/part_1',
        'batch_size': 2048,
        'benchmark_interval': 10,
        'test_interval': 10,
        'test_step': 32,
        'save_interval': 2000,
        'emb_size': 50,
        'vid_window_size': 20,
        'feature_num': 20+1,

        'num_parallel_preprocess': 30,

        'nn_size': [256, 128, 64, 1],
        'nn_stddev': [0.04, 0.07, 0.016, 0.016],
        "nn_bias_stddev": [0.04, 0.07, 0.016, 0.016],
        'activation': ['selu', 'selu', 'selu', 'selu'],

        'sparse_placement': '/GPU:0',

        'timeline': True,
        'timeline_step': [10, 20, 40],
    }

    try:
        start_training(config)
    except:
        print(traceback.format_exc())
