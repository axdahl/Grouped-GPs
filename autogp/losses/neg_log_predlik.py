import numpy as np
from . import loss

class NegLogPredDensity(loss.Loss):
    """
    Currently defined for Gaussian posterior predictive density 1D.
    ytrue: tf.placeholder - tensor (or vector) of type 'float'
    ypred: tuple (y_hat, variance(y_hat)), each tensor of type 'float'
    """
    def __init__(self, dout):
        loss.Loss.__init__(self,dout)

    def eval(self, ytrue, ypred):
        errors =  0.5 * np.log(2.0 * np.pi * ypred[1]) + ((ytrue - ypred[0]) ** 2) / (2.0 * ypred[1])
        return np.mean(errors)


    def get_name(self):
        return "NLPD"
