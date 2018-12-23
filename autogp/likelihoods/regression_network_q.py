import numpy as np
import tensorflow as tf

from . import likelihood
from .. import util


class RegressionNetwork(likelihood.Likelihood):
    """
    GPRN likelihood compatable with GaussianProcess class
    """
    def __init__(self, output_dim, latfunc_dim, std_dev, predict_samples=1000):
        self.latfunc_dim = latfunc_dim  # dim q
        self.output_dim = output_dim # dim p
        self.predict_samples = predict_samples
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

    def predict(self, latent_means, latent_vars):
        num_points = tf.shape(latent_means)[0]
        output_dims = tf.shape(latent_means)[1] # refers to number latent functions
        latent = (latent_means + tf.sqrt(latent_vars) *
                  tf.random_normal([self.predict_samples, num_points, output_dims]))
        pq = self.output_dim * self.latfunc_dim
        weights = latent[:, :, :pq] #weights up to p*q [:,:,q*p]
        inputs = latent[:, :, pq:]     # inputs p*q to end [:,:,q]
        split_weights = tf.stack(tf.split(weights, self.latfunc_dim, axis=2), axis=0)
        split_inputs = tf.stack(tf.split(inputs, self.latfunc_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0)

        pred_means = tf.reduce_mean(prod, 0)
        pred_vars = tf.reduce_sum((prod - pred_means) ** 2, 0) / (self.predict_samples - 1.0) + tf.exp(self.log_std_dev)
        return pred_means, pred_vars

