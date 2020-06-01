import tensorflow as tf

op = tf.strings.to_number("sfafafa")

sess = tf.Session()

res = sess.run(op)
print(op)
print(res)
