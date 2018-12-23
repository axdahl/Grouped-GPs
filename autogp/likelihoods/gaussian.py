import numpy as np
import tensorflow as tf

from . import likelihood


class Gaussian(likelihood.Likelihood):
    def __init__(self, std_dev=1.0):
        '''
        Gaussian likelihood implemented for P=1
        Compatable with GaussianProcess model class
        '''
        self.raw_std_dev = tf.Variable(std_dev)

    def log_cond_prob(self, outputs, latent):
        var = self.raw_std_dev ** 2
        return -0.5 * tf.log(2.0 * np.pi * var) - ((outputs - latent) ** 2) / (2.0 * var)

    def nlpd_cond_prob(self, outputs, latent):
        '''
        for P = 1 individual task-observation p(yip | fk) = p(yi | fk) = log_cond_prob
        '''
        var = self.raw_std_dev ** 2
        return -0.5 * tf.log(2.0 * np.pi * var) - ((outputs - latent) ** 2) / (2.0 * var)

    def get_params(self):
        return [self.raw_std_dev]

    def predict(self, latent_means, latent_vars):
        return latent_means, latent_vars + self.raw_std_dev ** 2

