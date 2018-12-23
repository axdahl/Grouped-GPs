import numpy as np
import tensorflow as tf

from . import likelihood
from .. import util


class RegressionNetworkLink(likelihood.Likelihood):
    """
    GPRN likelihood compatable with LinkGaussianProcess class
    """
    def __init__(self, output_dim, latfunc_dim, std_dev = 0.5):
        self.latfunc_dim = latfunc_dim  # dim q
        self.output_dim = output_dim # dim p
        self.log_std_dev = tf.Variable(np.ones([self.output_dim]) * np.log(std_dev),
                                       dtype=tf.float32)

    def log_cond_prob(self, outputs, latent):
        pq = self.output_dim * self.latfunc_dim
        weights = latent[:, :, :pq] #weights up to p*q [:,:,q*p]
        inputs = latent[:, :, pq:]     # inputs p*q to end [:,:,q]
        split_weights = tf.stack(tf.split(weights, self.latfunc_dim, axis=2), axis=0)
        split_inputs = tf.stack(tf.split(inputs, self.latfunc_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0)
        diff = outputs - prod 
        covar = tf.exp(self.log_std_dev)

        quad_form = tf.reduce_sum(1.0 / covar * (outputs - prod) ** 2, 2)
        #return -0.5 * (self.output_dim * tf.log(2.0 * np.pi) + tf.reduce_sum(covar) + quad_form)
        return -0.5 * (self.output_dim * tf.log(2.0 * np.pi) + tf.reduce_sum(self.log_std_dev) + quad_form)

    def nlpd_cond_prob(self, outputs, latent):
        # returns individual log probabilities for each sample for n,p i.e. [S, N, P]
        pq = self.output_dim * self.latfunc_dim
        weights = latent[:, :, :pq] #weights up to p*q [:,:,q*p]
        inputs = latent[:, :, pq:]     # inputs p*q to end [:,:,q]
        split_weights = tf.stack(tf.split(weights, self.latfunc_dim, axis=2), axis=0)
        split_inputs = tf.stack(tf.split(inputs, self.latfunc_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0)
        diff = outputs - prod
        covar = tf.exp(self.log_std_dev)
        
        return -0.5 * (tf.log(2.0 * np.pi) + self.log_std_dev + (1.0 / covar) * (outputs - prod) ** 2) 

    def get_params(self):
        return [self.log_std_dev]

    def predict(self, latent):        
        predict_samples_denom = tf.to_float(tf.shape(latent)[0]) - 1.0
        pq = self.output_dim * self.latfunc_dim
        weights = latent[:, :, :pq] #weights up to p*q [:,:,q*p]
        inputs = latent[:, :, pq:]     # inputs p*q to end [:,:,q]
        split_weights = tf.stack(tf.split(weights, self.latfunc_dim, axis=2), axis=0)
        split_inputs = tf.stack(tf.split(inputs, self.latfunc_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0)

        pred_means = tf.reduce_mean(prod, 0)
        pred_vars = tf.reduce_sum((prod - pred_means) ** 2, 0) / predict_samples_denom + tf.exp(self.log_std_dev)
        return pred_means, pred_vars

