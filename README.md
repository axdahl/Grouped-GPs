
# Grouped-GPs  (Author: Astrid Dahl)
This repo contains a generalised sparse variational inference class that implements dependent latent function models (as described in [Grouped Gaussian Processes for Solar Power Prediction](https://arxiv.org/abs/1806.02543)).
The package is based on AutoGP (see Acknowledgements) and adds the LinkGaussianProcess class for dependent latent GP models and augmented functionality for all classes including
- coregional and GPRN likelihood models for GaussianProcess and LinkGaussianProcess classes
- composite kernels and feature subset indexing
- generalised NLPD for non-Gaussian likelihoods
- additional flexible settings for optimisation
- print and reporting/retention options (predictions, nlpd, optimisation history, timing, other print reporting)

# Model Classes
LinkGaussianProcess:  
This is the main class implementing the GGP (Grouped Gaussian Process) model. Implemented for Kronecker-structured prior over latent functions in a group, and diagonal or full (Kronecker-structured) freely parameterised variational posterior. Note that this class incorporates sampling for prediction within the main class rather than in the likelihood function (as is done for GaussianProcess class). Compatible likelihood model is regression_network_linked.py.

GaussianProcess:  
Based on the original class from AutoGP, with independent latent functions, but augmented to also have the extended functionality of the LinkGaussianProcess class. Compatable likelihood models are coregion.py, regression_network_q.py (user defined q), gaussian.py.

# Requirements
Requires Python 3, Tensorflow 1.1-1.3
NB Adjust setup.py for compiler settings - currently hardcoded for gcc4 compatability to support triangular matrix ops code from GPflow (see acknowledgements).

# Acknowledgements
Grouped-GPs was originally based on AutoGP for Python 2 (cloned September 2017) - an implementation of the model described in [AutoGP: Exploring the Capabilities and Limitations of Gaussian Process Models](https://arxiv.org/abs/1610.05392).
The original AutoGP code can be found at (http://github.com/ebonilla/AutoGP).

AutoGP and Grouped-GPs both make use of code to support triangular matrix operations (under `autogp/util/tf_ops`) from the GPflow repository (Hensman, Matthews et al. GPflow, http://github.com/GPflow/GPflow, 2016). Grouped-GPs also adapts code from GPflow for several kernels.
