# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 10:54:18 2016

@author: adahl

Script to execute example ggp multisite forecasting model. 
Inputs: Data training and test sets (dictionary pickle), link data (ndarray pickle - same used for all groups)

Input order for model:
                 output_dim,
                 likelihood_func,
                 kernel_funcs,
                 link_kernel_funcs,
                 block_struct,
                 inducing_inputs,
                 link_inputs,
                 num_components=1,
                 diag_post=True,
                 num_samples=100,
                 predict_samples=1000

Where functions are independent i.e. in own block, set link_kernel[i] = link_inputs[i] = 1.0

Data for example: 
 - normalised solar data for 10 sites for 15 minute forecast
 - N_train = 3000, N_test = 2000, P = 10, D = 31
 - Xtr[:, :30] 3 recent lagged observations for each site in order
 - Xtr[:, 30] time index
 - link inputs is a 10x2 array (link inputs repeated for every group) 
   with normalised lat,long for each site in order

For faster testing, use random initialisation for inducing inputs instead of cluster centroids.

"""

import os
import numpy as np
import pickle
import pandas as pd
import traceback
import time
import sklearn.cluster
import csv
import sys

import autogp
from autogp import likelihoods
from autogp import kernels
import tensorflow as tf
from autogp import datasets
from autogp import losses
from autogp  import util


dpath = '/path/to/pickle/data'  # user update

dfile = 'example_gprn_inputsdict.pickle'
dlinkfile = 'example_gprn_linkinputsarray.pickle'

outdir = '/path/to/working/dir'  # user update

def get_inputs():
    """
    inputsdict contains {'Yte': Yte, 'Ytr': Ytr, 'Xtr': Xtr, 'Xte': Xte} where values are np.arrays
    np. arrays are truncated to evenly split into batches of size = 200

    returns inputsdict, Xtr_link (ndarray, shape = [P, D_link_features])
    """
    with open(os.path.join(dpath, dfile), 'rb') as f:
        d_all = pickle.load(f)

    with open(os.path.join(dpath, dlinkfile), 'rb') as f:
        d_link = pickle.load(f)

    return d_all, d_link


def init_z(train_inputs, num_inducing):
    # Initialize inducing points using clustering.
    mini_batch = sklearn.cluster.MiniBatchKMeans(num_inducing)
    cluster_indices = mini_batch.fit_predict(train_inputs)
    inducing_locations = mini_batch.cluster_centers_
    return inducing_locations


FLAGS = util.util.get_flags()
BATCH_SIZE = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
DISPLAY_STEP = FLAGS.display_step
EPOCHS = FLAGS.n_epochs
NUM_SAMPLES =  FLAGS.mc_train
NUM_INDUCING = FLAGS.n_inducing
NUM_COMPONENTS = FLAGS.num_components
IS_ARD = FLAGS.is_ard
TOL = FLAGS.opt_tol
VAR_STEPS = FLAGS.var_steps
DIAG_POST = FLAGS.diag_post
PRED_SAMPLES = FLAGS.pred_samples
NLPDS = FLAGS.save_nlpds

MAXTIME = 300

# define GPRN P and Q
output_dim = 6 #P
node_dim = 6    #Q
lag_dim = 3  # for parsing lag features for individual latent functions

# extract dataset
d, d_link = get_inputs()
Ytr, Yte, Xtr, Xte = d['Ytr'], d['Yte'], d['Xtr'], d['Xte']

data = datasets.DataSet(Xtr.astype(np.float32), Ytr.astype(np.float32), shuffle=False)
test = datasets.DataSet(Xte.astype(np.float32), Yte.astype(np.float32), shuffle=False)

print("dataset created")


# model config block rows (where P=Q): block all w.1, w.2 etc, leave f independent
# order of block_struct is rows, node functions
# lists required: block_struct, link_inputs, kern_link, kern

#block_struct nested list of grouping order
block_struct = [[] for _ in range(output_dim)]
for i in range(output_dim):
    row = list(range(i, i+output_dim*(node_dim-1)+1, output_dim))
    block_struct[i] = row

nodes = [[x] for x in list(range(output_dim * node_dim, output_dim * node_dim + output_dim))]
block_struct = block_struct + nodes

# link inputs used repeatedly but can have different link inputs
link_inputs = [d_link for i in range(output_dim)] + [1.0 for i in range(output_dim)] # for full row blocks, independent nodes

# create 'between' kernel list
klink_rows = [kernels.CompositeKernel('mul',[kernels.RadialBasis(2, std_dev=1.0, lengthscale=1.0, white=0.01, input_scaling = IS_ARD),
                                            kernels.CompactSlice(2, active_dims=[0,1], lengthscale = 1.0, input_scaling = IS_ARD)] )
                                            for i in range(output_dim) ]

klink_g = [1.0 for i in range(output_dim)]
kernlink = klink_rows +  klink_g

# create 'within' kernel list
# setup for example data - extract lag features for each site to use in associated node functions/blocks
lag_active_dims_s = [ [] for _ in range(output_dim)]
for i in range(output_dim):
    lag_active_dims_s[i] = list(range(lag_dim*i, lag_dim*(i+1)))

k_rows = [kernels.CompositeKernel('mul',[kernels.RadialBasisSlice(lag_dim, active_dims=lag_active_dims_s[i], 
                                            std_dev = 0.5, white = 0.01, input_scaling = IS_ARD),
                                            kernels.PeriodicSliceFixed(1, active_dims=[Xtr.shape[1]-1], lengthscale=1.0, 
                                            std_dev=0.5, period = 144) ])
                                            for i in range(output_dim)] 

k_g = [kernels.RadialBasisSlice(lag_dim, active_dims=lag_active_dims_s[i], 
                                            std_dev = 1.0, white = 0.01, input_scaling = IS_ARD)
                                            for i in range(output_dim)]

kern = k_rows + k_g

likelihood = likelihoods.RegressionNetworkLink(output_dim, node_dim, std_dev = 0.1)  # p, q, lik_noise

Z = init_z(data.X, NUM_INDUCING)
m = autogp.LinkGaussianProcess(output_dim, likelihood, kern, kernlink, block_struct, Z, link_inputs,
    num_components=NUM_COMPONENTS, diag_post=DIAG_POST, num_samples=NUM_SAMPLES, predict_samples=PRED_SAMPLES)


# initialise losses and logging
os.chdir(outdir)
error_rate = losses.RootMeanSqError(data.Dout)

with open("log_results.csv", 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'fit_runtime', 'nelbo', error_rate.get_name(),'generalised_nlpd'])
with open("log_params.csv", 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'raw_kernel_params', 'raw_kernlink_params', 'raw_likelihood_params', 'raw_weights'])
with open("log_comp_time.csv", 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'batch_time', 'nelbo_time', 'pred_time', 'gen_nlpd_time', error_rate.get_name()+'_time'])


# optimise
o = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.9,beta2=0.99)

print("start time = ", time.strftime('%X %x %Z'))
m.fit(data, o, var_steps = VAR_STEPS, epochs = EPOCHS, batch_size = BATCH_SIZE, display_step=DISPLAY_STEP, test = test,
        loss = error_rate, tolerance = TOL, max_time=MAXTIME )
print("optimisation complete")

# export final predicted values and loss metrics
ypred = m.predict(test.X, batch_size = BATCH_SIZE) #same batchsize used for convenience
np.savetxt("predictions.csv", np.concatenate(ypred, axis=1), delimiter=",")

# nlpd samples generate large files
if NLPDS == True:
    nlpd_samples, nlpd_meanvar = m.nlpd_samples(test.X, test.Y, batch_size = BATCH_SIZE) 
    try:
        np.savetxt("nlpd_meanvar.csv", nlpd_meanvar, delimiter=",")  # N x 2P as for predictions
    except:
        print('nlpd_meanvar export fail')
    try:
        np.savetxt("nlpd_samples.csv", nlpd_samples, delimiter=",")  # NP x S (NxS concat for P tasks)
    except:
        print('nlpd_samples export fail')


print("Final " + error_rate.get_name() + "=" + "%.4f" % error_rate.eval(test.Y, ypred[0]))
print("Final " + "generalised_nlpd" + "=" + "%.4f" % m.nlpd_general(test.X, test.Y, batch_size = BATCH_SIZE))
print("mean predictive variance: ", np.mean(ypred[1]))

print("finish time = " + time.strftime('%X %x %Z'))


