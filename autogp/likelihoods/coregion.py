import numpy as np
import tensorflow as tf

from . import likelihood
from .. import util


class Coregion(likelihood.Likelihood):
    """
    Compatable with GaussianProcess model class
    """
    def __init__(self, output_dim, latfunc_dim, std_dev, weight):
        self.latfunc_dim = latfunc_dim  # dim q
        self.output_dim = output_dim # dim p
        self.log_std_dev = tf.Variable(np.ones([self.output_dim]) * np.log(std_dev), dtype=tf.float32)
        self.weights = tf.Variable(np.ones([self.output_dim, self.latfunc_dim]) * weight, dtype = tf.float32)
        
    def log_cond_prob(self, outputs, latent):
       # slight hack - propogate weights [p,q] to split_weights [q,S,n,p]
        weights = tf.transpose(self.weights) #now [q,p]
        split_weights = tf.tile(tf.expand_dims(weights, axis = 1), multiples = [1,tf.shape(latent)[0],1])
        split_weights = tf.tile(tf.expand_dims(split_weights, axis = 2), multiples = [1,1,tf.shape(latent)[1],1])

        split_inputs = tf.stack(tf.split(latent, self.latfunc_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0)
        diff = outputs - prod # outputs=[:,p] 
        covar = tf.exp(self.log_std_dev)
        quad_form = tf.reduce_sum(1.0 / covar * (outputs - prod) ** 2, 2)
        #return -0.5 * (self.output_dim * tf.log(2.0 * np.pi) + tf.reduce_sum(covar) + quad_form)
        return -0.5 * (self.output_dim * tf.log(2.0 * np.pi) + tf.reduce_sum(self.log_std_dev) + quad_form)

    def nlpd_cond_prob(self, outputs, latent):
        # returns individual log probabilities for each sample for n,p i.e. [S, N, P]
        weights = tf.transpose(self.weights) #now [q,p]
        split_weights = tf.tile(tf.expand_dims(weights, axis = 1), multiples = [1,tf.shape(latent)[0],1])
        split_weights = tf.tile(tf.expand_dims(split_weights, axis = 2), multiples = [1,1,tf.shape(latent)[1],1])

        split_inputs = tf.stack(tf.split(latent, self.latfunc_dim, axis=2), axis=0)
        prod_split = split_weights * split_inputs # [q,:,:,p]
        prod = tf.reduce_sum(prod_split, axis=0)
        diff = outputs - prod # outputs=[:,p] 
        covar = tf.exp(self.log_std_dev)
        return -0.5 * (tf.log(2.0 * np.pi) + self.log_std_dev + (1.0 / covar) * (outputs - prod) ** 2) 

    def get_params(self):
        return [self.weights, self.log_std_dev]

    def predict(self, latent_means, latent_vars):
        # calculate means
        # latent_means and latent_vars are batch_n x q
        latent_means = tf.expand_dims(latent_means, axis=2) # n x q x 1
        # expand weights from pxq to nxpxq
        # and matmul inner dimensions weights, latent_means giving n x p x 1
        pred_means = tf.matmul(tf.tile(tf.expand_dims(self.weights, axis=0), multiples = [tf.shape(latent_means)[0],1,1]), 
                                latent_means)
        pred_means = tf.squeeze(pred_means, axis=2) #return to n x p

        # calculate variances
        weight_vars = self.weights ** 2
        latent_vars = tf.expand_dims(latent_vars, axis=2) # n x q x 1
        # expand weights from pxq to nxpxq
        # and matmul inner dimensions weights, latent_vars giving n x p x 1
        pred_vars = tf.matmul(tf.tile(tf.expand_dims(self.weights, axis=0), multiples = [tf.shape(latent_vars)[0],1,1]), 
                                latent_vars)
        pred_vars = tf.squeeze(pred_vars, axis=2) + tf.exp(self.log_std_dev)   #return to n x p

        return pred_means, pred_vars # n x p, n x p

