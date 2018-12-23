'''
Defines kernel functions that are composites
of other kernels.

Allows addition or multiplication of two components - for more use in heirarchical structure

Inputs: requires list of kernels (kerns, length==2) and math operation for combining them ('add' or 'mul')
'''

import numpy as np
import tensorflow as tf

from .. import util
from . import kernel
from .kernel_extras import *


class CompositeKernel(kernel.Kernel):

    def __init__(self, operation, kernel_inputs=None, white=0.01):
        self.kerns = kernel_inputs
        self.operation = operation
        self.white = white

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = self.white * util.eye(tf.shape(points1)[0]) + 0.1*self.white*tf.ones([tf.shape(points1)[0],tf.shape(points1)[0]])
        else:
            white_noise = 0.1*self.white

        if self.operation == 'add':
            kern = tf.add(self.kerns[0].kernel(points1, points2), self.kerns[1].kernel(points1, points2))
        elif self.operation == 'mul':
            kern = tf.multiply(self.kerns[0].kernel(points1, points2), self.kerns[1].kernel(points1, points2))
        return kern + white_noise

    def diag_kernel(self, points):
        if self.operation == 'add':
            kern = tf.add(self.kerns[0].diag_kernel(points), self.kerns[1].diag_kernel(points))
        if self.operation == 'mul':
            kern = tf.multiply(self.kerns[0].diag_kernel(points), self.kerns[1].diag_kernel(points))
        return kern + self.white

    def get_params(self):
        return self.kerns[0].get_params() + self.kerns[1].get_params()
