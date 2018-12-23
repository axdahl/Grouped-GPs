from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import csv

from . import kernels
from . import likelihoods
from . import util


class LinkGaussianProcess(object):
    """ 
    Main class representing the GGP model where latent functions may covary
        with Kronecker-structured prior covariance
        and diagonal or Kronecker-structure variational posterior covariance.
    This class is implemented for batch nelbo optimisation without LOO optimisation

    Parameters
    ----------
    likelihood_func : subclass of likelihoods.Likelihood
        An object representing the likelihood function p(y|f).
    kernel_funcs : list of subclasses of kernels.Kernel
        A list of one kernel per block of block of latent functions. len = R
    inducing_inputs : ndarray
        An array of initial inducing input locations. Dimensions: num_inducing * input_dim.
    num_components : int
        The number of mixture of Gaussian components.
    diag_post : bool
        True if the mixture of Gaussians uses a diagonal covariance, False otherwise.
    num_samples : int
        The number of samples to approximate the expected log likelihood of the posterior.
    predict_samples : int
        The number of samples to approximate expected latent function AND approximate
        negative log predictive density (nlpd)
    link_kernel_funcs : list of subclasses of kernels.Kernel
        A list of one kernel per block of latent functions. len = R
    block_struct : nested list of integers - latent function block groupings.
        For example: 6 latent functions grouped to 3 blocks as [[0,2,1],[4],[3,5]]

    link_inputs : list of R ndarrays
        each ndarray takes Qr x D_link feature set
        for independent latent functions (block size ==1) use dummy value 1.0
        For example: 6 latent functions grouped to 3 blocks as above requires [ndarray, 1.0, ndarray]

    """
    def __init__(self,
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
                 predict_samples=1000):

        self.likelihood = likelihood_func
        self.kernels = kernel_funcs
        self.kern_links = link_kernel_funcs
        # Save whether our posterior is diagonal or not.
        self.diag_post = diag_post

        # Repeat the inducing inputs for all latent blocks if we haven't been given individually
        # specified inputs per block.
        if inducing_inputs.ndim == 2:
            inducing_inputs = np.tile(inducing_inputs[np.newaxis, :, :], [len(self.kernels), 1, 1])

        # Initialize all model dimension constants.
        self.num_components = num_components
        self.num_latent = len([i for r in block_struct for i in r]) #flatten block_struct
        self.num_block = len(self.kern_links)
        self.ell_samples = num_samples
        self.num_inducing = inducing_inputs.shape[1]
        self.block_struct = block_struct 
        self.input_dim = inducing_inputs.shape[2]
        self.output_dim = output_dim
        self.predict_samples = predict_samples
        self.link_inputs = [tf.constant(x, dtype = tf.float32) for x in link_inputs]

        # Define all parameters that get optimized directly in raw form. Some parameters get
        # transformed internally to maintain certain pre-conditions.
        self.raw_weights = tf.Variable(tf.zeros([self.num_components]))
        self.raw_means = tf.Variable(tf.zeros([self.num_components, self.num_latent,
                                               self.num_inducing]))
        if self.diag_post:
            self.raw_covars = tf.Variable(tf.ones([self.num_components, self.num_latent,
                                                   self.num_inducing]))
            self.raw_link_covars = tf.Variable(tf.zeros([1]), trainable = False)

        else:
            init_vec = np.zeros([self.num_components, self.num_block] +
                                 [int(x) for x in util.tri_vec_shape(self.num_inducing)], dtype=np.float32) 
            self.raw_covars = tf.Variable(init_vec)
            # create raw_link_covars
            init_linkvec = np.zeros([self.num_components, self.num_block] + 
                                [int(x) for x in util.tri_vec_shape(len(max(self.block_struct, key=len)))], dtype=np.float32)
            self.raw_link_covars = tf.Variable(init_linkvec)



        

        self.raw_inducing_inputs = tf.Variable(inducing_inputs, dtype=tf.float32)
        self.raw_likelihood_params = self.likelihood.get_params()
        self.raw_kernel_params = sum([k.get_params() for k in self.kernels], [])
        self.raw_kernlink_params = sum([k.get_params() for k in self.kern_links if type(k) is not float], [])
    
        # Define placeholder variables for training and predicting.
        self.num_train = tf.placeholder(tf.float32, shape=[], name="num_train")
        self.train_inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim],
                                           name="train_inputs")
        self.train_outputs = tf.placeholder(tf.float32, shape=[None, None],
                                            name="train_outputs")
        self.test_inputs = tf.placeholder(tf.float32, shape=[None, self.input_dim],
                                          name="test_inputs")
        self.test_outputs = tf.placeholder(tf.float32, shape=[None, None],
                                          name="test_outputs")

        # Now build our computational graph.
        self.nelbo, self.predictions, self.general_nlpd = self._build_graph(self.raw_weights,
                                                                        self.raw_means,
                                                                        self.raw_covars,
                                                                        self.raw_link_covars,
                                                                        self.raw_inducing_inputs,
                                                                        self.train_inputs,
                                                                        self.train_outputs,
                                                                        self.num_train,
                                                                        self.test_inputs,
                                                                        self.test_outputs)

        # Do all the tensorflow bookkeeping.
        self.session = tf.Session(config=tf.ConfigProto(operation_timeout_in_ms = 60000))
        self.optimizer = None
        self.train_step = None

    def fit(self, data, optimizer, var_steps=10, epochs=200,
            batch_size=200, display_step=1, test=None, loss=None, tolerance = None, max_time=300):
        """ 
        Fit the Gaussian process model to the given data.

        Parameters
        ----------
        data : subclass of datasets.DataSet
            The train inputs and outputs.
        optimizer : TensorFlow optimizer
            The optimizer to use in the fitting process.
        loo_steps : int
            Number of steps  to update hyper-parameters using loo objective
            NB LOO OPTIMISATION DISABLED
        var_steps : int
            Number of steps to update  variational parameters using variational objective (elbo).
        epochs : int
            The number of epochs to optimize the model for.
        batch_size : int
            The number of datapoints to use per mini-batch when training. If batch_size is None,
            then we perform batch gradient descent.
        display_step : int
            The frequency at which the objective values are printed out.
        tolerance : float
            Convergence criterion relative change in nelbo over successive epoch
        max_time : int
            Maximum fit runtime
        """
        num_train = data.num_examples
        if batch_size is None:
            batch_size = num_train

        if self.optimizer != optimizer:
            self.optimizer = optimizer
            self.train_step = optimizer.minimize(
                self.nelbo, var_list=[self.raw_inducing_inputs] +
                                        self.raw_kernel_params +
                                        self.raw_kernlink_params +
                                        self.raw_likelihood_params +
                                        [self.raw_weights] +
                                        [self.raw_means] +
                                        [self.raw_covars] +
                                        [self.raw_link_covars] # diag post can comment out
                                        )
            self.session.run(tf.global_variables_initializer())

        # export start values
        kout=[repr(data.epochs_completed), [np.concatenate(self.session.run(self.raw_kernel_params)).ravel()]]
        kout.append([self.session.run(self.raw_likelihood_params)])
        kout.append([self.session.run(self.raw_weights)])
        with open("log_params.csv", 'a', newline='') as f:
            csv.writer(f).writerow(kout)

        # initialise saver
        saver = tf.train.Saver([self.raw_inducing_inputs] +
                                self.raw_kernel_params +
                                self.raw_likelihood_params +
                                [self.raw_weights] +
                                [self.raw_means] +
                                [self.raw_covars] +
                                [self.raw_link_covars], 
                                max_to_keep = 1, save_relative_paths = True)

        start = data.next_batch(batch_size)

        old_epoch = 0
        old_nelbo = None
        stop_condition = False
        fit_stime = time.time()

        while data.epochs_completed < epochs:
            if stop_condition == True:
                break
            num_epochs = data.epochs_completed + var_steps
            batch_counter = 0
            while data.epochs_completed < num_epochs:
                if stop_condition == True:
                    break
                batch_stime = time.time()
                batch = data.next_batch(batch_size)
                print('current epoch =     ',data.epochs_completed)
                self.session.run(self.train_step, feed_dict={self.train_inputs: batch[0],
                                                             self.train_outputs: batch[1],
                                                             self.num_train: num_train})
                if data.epochs_completed % display_step == 0 and data.epochs_completed != old_epoch:
                    
                    new_nelbo = self._print_state(data, test, loss, num_train, 100, fit_stime, batch_stime) #batchsize set=100
                    saver.save(self.session, 'tf_saver_variablelog')

                    if old_nelbo:
                         stop_condition = self._stop_eval(old_nelbo, new_nelbo, tolerance, fit_stime, max_time)

                    old_epoch = data.epochs_completed
                    old_nelbo = new_nelbo
                batch_counter += 1
                print('batch_counter = ',batch_counter)


    def predict(self, test_inputs, batch_size=100):
        """ 
        Predict outputs given inputs.

        Parameters
        ----------
        test_inputs : ndarray
            Points on which we wish to make predictions. Dimensions: num_test * input_dim.
        batch_size : int
            The size of the batches we make predictions on. If batch_size is None, predict on the
            entire test set at once.

        Returns
        -------
        ndarray
            The predicted mean of the test inputs. Dimensions: num_test x output_dim.
        ndarray
            The predicted variance of the test inputs. Dimensions: num_test x output_dim.
        """
        if batch_size is None:
            num_batches = 1
        else:
            num_batches = int(util.ceil_divide(test_inputs.shape[0], batch_size))

        test_inputs = np.array_split(test_inputs, num_batches)
        pred_means = util.init_list(0.0, [num_batches])
        pred_vars = util.init_list(0.0, [num_batches])
        for i in range(num_batches):
            pred_means[i], pred_vars[i] = self.session.run(
                self.predictions, feed_dict={self.test_inputs: test_inputs[i]})

        return np.concatenate(pred_means, axis=0), np.concatenate(pred_vars, axis=0)

    def nlpd_general(self, test_inputs, test_outputs, batch_size=100):
        """ 
        Estimate negative log predictive density

        Parameters
        ----------
        test_inputs : ndarray
            Points on which we wish to make predictions. Dimensions: num_test x input_dim.
         test_outputs : ndarray
            Points on which we wish to evaluate log predictive density. Dimensions: num_test x input_dim.           
        batch_size : int
            The size of the batches we make predictions on. If batch_size is None, predict on the
            entire test set at once.

        Returns
        -------
        float
            Estimated negative log predictive density

        """
        num_test = test_outputs.shape[0]
        dim_out = test_outputs.shape[1]
        if batch_size is None:
            num_batches = 1
        else:
            num_batches = int(util.ceil_divide(num_test, batch_size))

        test_inputs = np.array_split(test_inputs, num_batches)
        test_outputs = np.array_split(test_outputs, num_batches)
        nlpds = util.init_list(0.0, [num_batches])
        for i in range(num_batches):
            nlpds[i] = self.session.run(
                self.general_nlpd, feed_dict={self.test_inputs: test_inputs[i],
                                                self.test_outputs: test_outputs[i]})
        return np.sum([np.sum(x) for x in nlpds])/(num_test * dim_out * self.predict_samples)

    def nlpd_samples(self, test_inputs, test_outputs, batch_size=100):
        """ 
        Estimate negative log predictive density at each sample point

        Parameters
        ----------
        test_inputs : ndarray
            Points on which we wish to make predictions. Dimensions: num_test * input_dim.
         test_outputs : ndarray
            Points on which we wish to evaluate log predictive density. Dimensions: num_test * input_dim.           
        batch_size : int
            The size of the batches we make predictions on. If batch_size is None, predict on the
            entire test set at once.

        Returns
        -------

        tensor [S,N,P] containing negative log predictive density for each task/observation for S samples
        from posterior q(f)_nk

        """
        num_test = test_outputs.shape[0]
        dim_out = test_outputs.shape[1]
        if batch_size is None:
            num_batches = 1
        else:
            num_batches = int(util.ceil_divide(num_test, batch_size))

        test_inputs = np.array_split(test_inputs, num_batches)
        test_outputs = np.array_split(test_outputs, num_batches)
        nlpds = util.init_list(0.0, [num_batches])
        for i in range(num_batches):
            nlpds[i] = self.session.run(
                self.general_nlpd, feed_dict={self.test_inputs: test_inputs[i],
                                                self.test_outputs: test_outputs[i]})

        # concat batches, calculate sample mean, var for each observation
        nlpds = np.concatenate(nlpds, axis=1) # SxNxP
        nlpds_meanvar = np.concatenate([np.mean(nlpds, axis=0), np.var(nlpds, axis=0)], axis=1) # Nx2P
        nlpds = np.transpose(nlpds) # SxNxP --> PxNxS
        nlpds = np.concatenate([np.squeeze(x, axis=0) 
            for x in np.split(nlpds, self.output_dim, axis = 0)], axis=0) # NP x S (tasks stacked)

        return nlpds, nlpds_meanvar


    def _print_state(self, data, test, loss, num_train, batch_size, fit_stime, batch_stime):
        batch_time = round((time.time() - batch_stime),3)
        fit_runtime = round((time.time() - fit_stime)/60,4)
        print("batch runtime: ", batch_time, " sec")
        print("fit runtime: ", fit_runtime, " min")
        nelbo_stime = time.time()
        if num_train <= 100000 or batch_size is not None:
            num_batches = round(num_train/batch_size) 
            nelbo_inputs = np.array_split(data.X, num_batches)
            nelbo_outputs = np.array_split(data.Y, num_batches)
            nelbo_batches = util.init_list(0.0, [num_batches])
            for i in range (num_batches):
                nelbo_batches[i] = self.session.run(self.nelbo, feed_dict={self.train_inputs: nelbo_inputs[i],
                                                                self.train_outputs: nelbo_outputs[i],
                                                                self.num_train: num_train})
            nelbo = sum(nelbo_batches)
            nelbo_time = round((time.time() - nelbo_stime),3)
            print('nelbo computation time: ', nelbo_time, " sec.")
            print("i=" + repr(data.epochs_completed) + " nelbo=" + repr(nelbo), end=" ")

        # get losses
        if loss is not None:
            # predictions
            pred_stime = time.time()
            ypred = self.predict(test.X, batch_size=100)
            pred_time = round((time.time() - pred_stime),3)
            print('predictions computation time: ', pred_time, " sec.")

            # gen nlpd
            nlpd_stime = time.time()
            gen_nlpd = self.nlpd_general(test.X, test.Y, batch_size=100)
            nlpd_time = round((time.time() - nlpd_stime),3)
            print('gen nlpd computation time: ', nlpd_time, " sec.") 

            # other loss
            loss_stime = time.time()
            if loss.get_name() == 'NLPD':
                loss_update = loss.eval(test.Y, ypred)
            else:
                loss_update = loss.eval(test.Y, ypred[0])
            loss_time = round((time.time() - loss_stime), 3)
            
            print("i=" + repr(data.epochs_completed) + " current " + loss.get_name() + "=" + "%.4f" % loss_update)
            print("i=" + repr(data.epochs_completed) + " current generalised nlpd =" + "%.4f" % gen_nlpd)

            # append logs
            with open("log_results.csv", 'a', newline='') as f:
                csv.writer(f).writerow([repr(data.epochs_completed), fit_runtime, nelbo, loss_update, gen_nlpd])

            kout = [repr(data.epochs_completed), batch_time, nelbo_time, pred_time, nlpd_time, loss_time]
            with open("log_comp_time.csv", 'a', newline='') as f:
                csv.writer(f).writerow(kout)

        # export parameters and predictions
        kout=[repr(data.epochs_completed), [np.concatenate(self.session.run(self.raw_kernel_params)).ravel()]]
        if self.raw_kernlink_params:
            kout.append([np.concatenate(self.session.run(self.raw_kernlink_params)).ravel()])
        kout.append([self.session.run(self.raw_likelihood_params)])
        kout.append([self.session.run(self.raw_weights)])
        with open("log_params.csv", 'a', newline='') as f:
            csv.writer(f).writerow(kout)
        np.savetxt("predictions.csv", np.concatenate(ypred, axis=1), delimiter=",")

        return nelbo

    def _stop_eval(self, old_nelbo, new_nelbo, tolerance, fit_stime, max_time):
        fit_time = round((time.time() - fit_stime)/60,4)
        d = (new_nelbo - old_nelbo)/old_nelbo
        print("proportion change in nelbo ", round(d,7))
        if abs(d) < tolerance or fit_time >= max_time:
            return True
        else:
            return False

    def _build_graph(self, raw_weights, raw_means, raw_covars, raw_link_covars, raw_inducing_inputs,
                     train_inputs, train_outputs, num_train, test_inputs, test_outputs):
        # normalise weights
        weights = tf.exp(raw_weights) / tf.reduce_sum(tf.exp(raw_weights))

        if self.diag_post:
            covars = tf.exp(raw_covars)
            link_covars = None
         
        else:
            covars_list = [None] * self.num_components
            for i in range(self.num_components):
                mat = util.vec_to_tri(raw_covars[i, :, :]) #creates mats by row ie r so RxMxM
                diag_mat = tf.matrix_diag(tf.matrix_diag_part(mat))
                exp_diag_mat = tf.matrix_diag(tf.exp(tf.matrix_diag_part(mat)))
                covars_list[i] = mat - diag_mat + exp_diag_mat
            covars = tf.stack(covars_list, 0)

            # create nested list of posterior link parameters
            link_covars = [None] * self.num_components
            for i in range(self.num_components):
                mat = util.vec_to_tri(raw_link_covars[i, :, :]) #creates mats by row ie r so R x max(Qr) x max(Qr)
                diag_mat = tf.matrix_diag(tf.matrix_diag_part(mat))
                exp_diag_mat = tf.matrix_diag(tf.exp(tf.matrix_diag_part(mat)))
                mats_in = mat - diag_mat + exp_diag_mat # R x max(Qr) x max(Qr)

                # trim ragged block sizes and retain as list
                mats_in = tf.unstack(mats_in, axis=0) # split into R mats shaped max(Qr) x max(Qr)
                for r in range(self.num_block):
                    if len(self.block_struct[r]) == 1:  # keep dims where trimmed to scalar
                        mats_in[r] = tf.expand_dims(tf.expand_dims(mats_in[r][0, 0], axis=0), axis=1)
                    else:
                        mats_in[r] = mats_in[r][:len(self.block_struct[r]), :len(self.block_struct[r])]

                link_covars[i] = mats_in
                                    
        
        # Both inducing inputs and the posterior means can vary freely so don't change them.
        means = raw_means
        inducing_inputs = raw_inducing_inputs

        # Build the matrices of covariances between inducing inputs.

        kernel_mat = [self.kernels[r].kernel(inducing_inputs[r, :, :])
                      for r in range(self.num_block)] 
        kernel_chol = [tf.cholesky(k) for k in kernel_mat]

        # generate K(j,j') for each block of latent functions
        # where dim (block) = 1 (i.e. independent latent function), mat/chol set == 1
        kernlink_mat = util.init_list(1.0, [len(self.kern_links)])
        kernlink_chol = util.init_list(1.0, [len(self.kern_links)])
        for r in range(len(self.kern_links)):
            if self.kern_links[r] == 1.0:     # flag value from model input
                continue
            else:
                kernlink_mat[r] = self.kern_links[r].kernel(self.link_inputs[r])
                kernlink_chol[r] = tf.cholesky(kernlink_mat[r])


        # Now build the objective function.
        entropy = self._build_entropy(weights, means, covars, link_covars)
        cross_ent = self._build_cross_ent(weights, means, covars, link_covars, kernel_chol, kernlink_chol)
        ell = self._build_ell(weights, means, covars, link_covars, inducing_inputs,
                              kernel_chol, kernlink_chol, kernlink_mat, train_inputs, train_outputs)
        batch_size = tf.to_float(tf.shape(train_inputs)[0])
        nelbo = -((batch_size / num_train) * (entropy + cross_ent) + ell)

        # Finally, build the prediction function.
        predictions = self._build_predict(weights, means, covars, link_covars, inducing_inputs,
                                          kernel_chol, kernlink_chol, kernlink_mat, test_inputs)
        # Build the nlpd function.
        general_nlpd = self._build_nlpd(weights, means, covars, link_covars, inducing_inputs,
                                        kernel_chol, kernlink_chol, kernlink_mat, test_inputs, test_outputs)
        return nelbo, predictions, general_nlpd


    def _build_predict(self, weights, means, covars, link_covars, inducing_inputs,
                       kernel_chol, kernlink_chol, kernlink_mat, test_inputs):
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, kernlink_chol,
                                                        kernlink_mat,
                                                        inducing_inputs, test_inputs)
        pred_means = util.init_list(0.0, [self.num_components])
        pred_vars = util.init_list(0.0, [self.num_components])
        for i in range(self.num_components):
            covar_input = covars[i, :, :] if self.diag_post else covars[i, :, :, :]
            link_cov_input = None if self.diag_post else link_covars[i] # list of R covar link mats (each Qr x Qr)
            # Note (Ast) generate f|lambda distribution parameters
            latent_samples = self._build_samples(kern_prods, kern_sums,
                                                 means[i, :, :], covar_input, link_cov_input, self.predict_samples)
            # reorder latent according to 'inverted' block struct order
            latent_j = [j for r in self.block_struct for j in r] #implicit order of j in latent_samples
            revert_j = tf.invert_permutation(latent_j)
            latent_samples = tf.gather(latent_samples, revert_j, axis=2) # reorder to j=1, j=2, ...
            # Note (Ast) generate predicted y = Wf
            pred_means[i], pred_vars[i] = self.likelihood.predict(latent_samples)

        pred_means = tf.stack(pred_means, 0)
        pred_vars = tf.stack(pred_vars, 0)

        # Compute the mean and variance of the gaussian mixture from their components.
        weights = tf.expand_dims(tf.expand_dims(weights, 1), 1)
        weighted_means = tf.reduce_sum(weights * pred_means, 0)
        weighted_vars = (tf.reduce_sum(weights * (pred_means ** 2 + pred_vars), 0) -
                         tf.reduce_sum(weights * pred_means, 0) ** 2)

        return weighted_means, weighted_vars

    def _build_nlpd(self, weights, means, covars, link_covars, inducing_inputs,
                       kernel_chol, kernlink_chol, kernlink_mat, test_inputs, test_outputs):
        '''
        returns  -lpd_all (tensor for all n,p,s)
        '''
        lpd_all = tf.zeros([self.predict_samples, tf.shape(test_inputs)[0], self.output_dim])
        #lpd = 0
        dim_out = self.output_dim
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, kernlink_chol,
                                                        kernlink_mat,
                                                        inducing_inputs, test_inputs)
        for i in range(self.num_components):
            covar_input = covars[i, :, :] if self.diag_post else covars[i, :, :, :]
            link_cov_input = None if self.diag_post else link_covars[i] # list of R covar link mats (each Qr x Qr)
            latent_samples = self._build_samples(kern_prods, kern_sums,
                                                means[i, :, :], covar_input, link_cov_input, self.predict_samples)
            # reorder latent according to 'inverted' block struct order
            latent_j = [j for b in self.block_struct for j in b] #implicit order of j in latent_samples
            revert_j = tf.invert_permutation(latent_j)
            latent_samples = tf.gather(latent_samples, revert_j, axis=2) # reorder to j=1, j=2, ...

            lpd_all += weights[i] * self.likelihood.nlpd_cond_prob(test_outputs, latent_samples)

        return -lpd_all

    def _build_entropy(self, weights, means, covars, link_covars):
        # First build half a square matrix of normals. This avoids re-computing symmetric normals.
        log_normal_probs = util.init_list(0.0, [self.num_components, self.num_components])
        for i in range(self.num_components):
            for j in range(i, self.num_components):
                # TODO (Ast) non diag: loop over r instead of k; new means, covars objects variable dims
                # need diag_post or kron(S) structure
                if self.diag_post:
                    for k in range(self.num_latent):   
                        normal = util.DiagNormal(means[i, k, :], covars[i, k, :] +
                                                                 covars[j, k, :])               
                        log_normal_probs[i][j] += normal.log_prob(means[j, k, :])

                else:
                    for r in range(self.num_block):
                        if i == j:
                            # Compute chol(2S) = sqrt(2)*chol(S).
                            chol_s = util.kronecker_mul(link_covars[i][r], covars[i, r, :, :])
                            covars_sum = tf.sqrt(2.0) * chol_s

                        else:
                            chol_i = util.kronecker_mul(link_covars[i][r], covars[i, r, :, :])
                            chol_j = util.kronecker_mul(link_covars[j][r], covars[j, r, :, :])
                            covars_sum = tf.cholesky(chol_i + chol_j)

                        mean_i = tf.concat([means[i, k, :] for k in self.block_struct[r]], axis=0)
                        mean_j = tf.concat([means[j, k, :] for k in self.block_struct[r]], axis=0)
                        normal = util.CholNormal(mean_i, covars_sum)
                        log_normal_probs[i][j] += normal.log_prob(mean_j)

        # Now compute the entropy.
        entropy = 0.0
        for i in range(self.num_components):
            weighted_log_probs = util.init_list(0.0, [self.num_components])
            for j in range(self.num_components):
                if i <= j:
                    weighted_log_probs[j] = tf.log(weights[j]) + log_normal_probs[i][j]
                else:
                    weighted_log_probs[j] = tf.log(weights[j]) + log_normal_probs[j][i]

            entropy -= weights[i] * util.logsumexp(tf.stack(weighted_log_probs))

        return entropy
        
    def _build_cross_ent(self, weights, means, covars, link_covars, kernel_chol, kernlink_chol):
        # TODO change of full covar_r definition
        cross_ent = 0.0
        for i in range(self.num_components):
            sum_val = 0.0
            for r in range(self.num_block):
                # construct chol(Kjj kron Kzz)
                block_chol_r = util.kronecker_mul(kernlink_chol[r], kernel_chol[r])
                if self.diag_post:
                    # construct block diag_post using link_covars
                    covars_r = tf.diag(tf.concat([covars[i,j,:] for j in self.block_struct[r]], axis=0))
                    trace = tf.trace(tf.cholesky_solve(block_chol_r, covars_r))
                 
                else:
                    covars_r = util.kronecker_mul(link_covars[i][r], covars[i, r, :, :])
                    trace = tf.reduce_sum(util.diag_mul(
                                tf.cholesky_solve(block_chol_r, covars_r),
                                tf.transpose(covars_r)))
                
                mean_r = tf.concat([means[i,j,:] for j in self.block_struct[r]], axis=0)
                sum_val += (util.CholNormal(mean_r, block_chol_r).log_prob(0.0) -
                            0.5 * trace)

            cross_ent += weights[i] * sum_val

        return cross_ent

    def _build_ell(self, weights, means, covars, link_covars, inducing_inputs,
                   kernel_chol, kernlink_chol, kernlink_mat, train_inputs, train_outputs):
        kern_prods, kern_sums = self._build_interim_vals(kernel_chol, kernlink_chol,
                                                        kernlink_mat,
                                                        inducing_inputs, train_inputs)
        ell = 0
        for i in range(self.num_components):
            covar_input = covars[i, :, :] if self.diag_post else covars[i, :, :, :]
            link_cov_input = None if self.diag_post else link_covars[i] # list of R covar link mats (each Qr x Qr)
            latent_samples = self._build_samples(kern_prods, kern_sums,
                                                 means[i, :, :], covar_input, link_cov_input, self.ell_samples)
            # reorder latent according to 'inverted' block struct order
            latent_j = [j for b in self.block_struct for j in b] #implicit order of j in latent_samples
            revert_j = tf.invert_permutation(latent_j)
            latent_samples = tf.gather(latent_samples, revert_j, axis=2) # reorder to j=1, j=2, ...

            ell += weights[i] * tf.reduce_sum(self.likelihood.log_cond_prob(train_outputs,
                                                                            latent_samples))

        return ell / self.ell_samples

    def _build_interim_vals(self, kernel_chol, kernlink_chol, 
                            kernlink_mat, inducing_inputs, train_inputs):
        kern_prods = util.init_list(0.0, [self.num_block])
        kern_sums = util.init_list(0.0, [self.num_block])
        for r in range(self.num_block):
            ind_train_kern = util.kronecker_mul(kernlink_mat[r], 
                            self.kernels[r].kernel(inducing_inputs[r, :, :], train_inputs))

            # Compute A = Kfu.Kuu^(-1) = (Kuu^(-1).Kuf)^T.
            block_chol = util.kronecker_mul(kernlink_chol[r], kernel_chol[r])
            kern_prods[r] = tf.transpose(tf.cholesky_solve(block_chol, ind_train_kern))

            # Compute Kff - AKuf
            kern_sums[r] = util.kronecker_mul(kernlink_mat[r], 
                            self.kernels[r].kernel(train_inputs)) - tf.matmul(kern_prods[r], ind_train_kern)
            
        return kern_prods, kern_sums

    def _build_samples(self, kern_prods, kern_sums, means, covars, link_covars, num_samples):
        sample_means, sample_vars = self._build_sample_info(kern_prods, kern_sums, means, covars, link_covars)
        
        block_samples = util.init_list(0.0, [self.num_block])
        for r in range(self.num_block):
            sample_means_r = sample_means[r]
            sample_chols_r = tf.tile(tf.expand_dims(sample_vars[r], axis = 1), multiples=[1,num_samples,1,1])
            batch_size = tf.shape(sample_means_r)[0]
            raw_norm = tf.random_normal([batch_size, num_samples, len(self.block_struct[r]), 1]) # shape = [N,S,Q_r,1]
            # scale raw samples - premultiply (Q_rx1) by chol(Q_rxQ_r) for each N,S - and return to shape = [S,N,Q_r]
            block_samples[r] = tf.squeeze(sample_means_r, axis=2) + tf.transpose(tf.squeeze(
                                                            tf.matmul(sample_chols_r, raw_norm), axis=3), perm=[1,0,2])

        return tf.concat(block_samples, axis=2)


        
    def _build_sample_info(self, kern_prods, kern_sums, means, covars, link_covars):
        # Generate posterior mean and covariance matrix for each block R.
        post_means = util.init_list(0.0, [self.num_block])
        post_vars = util.init_list(0.0, [self.num_block])
        r_n_submeans = util.init_list(0.0, [self.num_block])
        sample_means = util.init_list(0.0, [self.num_block])
        for r in range(self.num_block):
            if self.diag_post:
                covars_r = tf.diag(tf.concat([covars[j,:] for j in self.block_struct[r]], axis=0))
                quad_form = tf.matmul(tf.matmul(kern_prods[r], covars_r), kern_prods[r], transpose_b = True)
                
            else:
                covars_r_chol = util.kronecker_mul(link_covars[r], covars[r,: ,:])
                covars_r = tf.matmul(covars_r_chol, covars_r_chol, transpose_b = True)
                quad_form = tf.matmul(tf.matmul(kern_prods[r], covars_r), tf.transpose(kern_prods[r]))
            

            post_means[r] = tf.matmul(kern_prods[r], 
                            tf.concat([tf.expand_dims(means[j, :], 1) for j in self.block_struct[r]], axis=0))
            post_vars[r] = kern_sums[r] + quad_form

            # Construct sampling mean and covariance b_kn, COV_kn for each n

            # gather and concat m_n for all n in m in post_means
            # split block mean vector into latent function mean vectors for each block r #returns nested list
            r_n_submeans[r] = tf.split(post_means[r], len(self.block_struct[r]))
            # change dims and concat into n x qr matrix of means for each block
            sample_means[r] = tf.concat([tf.expand_dims(j,axis=1) for j in r_n_submeans[r]], axis = 1)


        # gather covar submatrices for n for each r, combine to make N jxj block diag matrices
        sample_vars = util.init_list(0.0, [self.num_block])
        for r in range(self.num_block):
            # extract submatrices j of block r in row order [[j11, j12,...,], [j21, j22,...], etc]
            row_submats = tf.split(post_vars[r], len(self.block_struct[r]), axis=0)
            submats_r = util.init_list(0.0, [len(self.block_struct[r])])
            for j in range(len(row_submats)):
                submats_r[j] = tf.split(row_submats[j], len(self.block_struct[r]), axis = 1)
            subdiags_r = util.init_list(0.0, [ len(self.block_struct[r]), len(self.block_struct[r]) ])
            for i in range(len(self.block_struct[r])):
                for j in range(len(self.block_struct[r])):
                    subdiags_r[i][j] = tf.expand_dims(tf.diag_part(submats_r[i][j]), axis = 0)
                subdiags_r[i] = tf.concat([subdiags_r[i][k] for k in range(len(self.block_struct[r]))], axis = 0) # q_r x n
            sample_vars[r] = tf.stack(subdiags_r, axis = 0) # q_r x q_r x n
            sample_vars[r] = tf.transpose(sample_vars[r], perm=[2,0,1]) # q_r x q_r x n to n x q_r x q_r

            jitter_diag = 0.001*tf.diag( tf.ones([ len(self.block_struct[r]) ]))
            jitter_full = 0.0001*tf.ones( [len(self.block_struct[r]), len(self.block_struct[r])] )

            sample_vars[r] = tf.cholesky(sample_vars[r] + jitter_diag + jitter_full)

        # sample_means = N x Q, sample_vars = N x Q x Q
        # returns means and covars implicitly in block_struct order (Q = q_r, q_r, ...) and q_r = r_j_i1, r_j_i2, ...
        # where i1, i2 etc are order in range(len(block_struct[r]))
        return sample_means, sample_vars    

