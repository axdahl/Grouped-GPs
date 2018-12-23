'''
Constant kernel

'''

import numpy as np
import tensorflow as tf

from .. import util
from . import kernel


class Constant(kernel.Kernel):

    def __init__(self, std_dev=1.0, white=0.01):
        self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        self.white = white

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0])
        else:
            white_noise = 0.0 #0.1 * self.white * tf.random_uniform( [tf.shape(points1)[0], tf.shape(points2)[0]] )

        return tf.exp(self.std_dev) * tf.ones([tf.shape(points1)[0], tf.shape(points2)[0]]) + white_noise


    def diag_kernel(self, points):
        return (tf.exp(self.std_dev) + self.white) * tf.ones([tf.shape(points)[0]])

    def get_params(self):
        return [self.std_dev]
