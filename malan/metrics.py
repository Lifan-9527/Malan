import sklearn as skl
import tensorflow as tf

def auc(sess, num_steps):
    sklearn.metrics.roc_auc_score(np.asarray(_lb), np.asarray(_pred))
    pass