import tensorflow as tf
import rest


class ModelBase(object):
    def __init__(self):
        """
        """
        pass

    def build_graph(self, io):
        """
        Pending to be overrided.
        :return: labels, predict, loss
        """
        pass
