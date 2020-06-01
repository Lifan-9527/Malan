import tensorflow as tf
import numpy as np
# using a placeholder
x = tf.placeholder(tf.float32, shape=[None,2])
dataset = tf.data.Dataset.from_tensor_slices(x)
data = np.random.sample((100,2))
iter = dataset.make_initializable_iterator() # create the iterator
el = iter.get_next()
with tf.Session() as sess:
    # feed the placeholder with data
    sess.run(iter.initializer, feed_dict={ x: data })
    print(sess.run(el)) # output [ 0.52374458  0.71968478]
