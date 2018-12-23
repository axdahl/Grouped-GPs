'''
Code based on GPflow to allow kernels to operate on subsets of features
'''
import tensorflow as tf

def dim_slice(self, points1, points2=None):
    '''
    _slice called by kernel function.
    slice correct dimensions for use in kernel
    as given by self.active_dims

    inputs matrices of input features NxD and/or MxD
    returns sliced matrices [:, self.input_dim]
    '''
    if isinstance(self.active_dims, slice):
        points1 = points1[:, self.active_dims]
        if points2 is not None:
            points2 = points2[:, self.active_dims]
        else:
            points2 = points1

    else: 
        # implies active_dims is iterable of integers
        points1 = tf.transpose(tf.gather(tf.transpose(points1), self.active_dims)) #tf 1.1 gather on axis=0 only

        if points2 is not None:
            points2 = tf.transpose(tf.gather(tf.transpose(points2), self.active_dims))
        else:
            points2 = points1

    return (points1, points2)


def dim_slice_diag(self, points):
    if isinstance(self.active_dims, slice):
        points = points[:, self.active_dims]
    else:
        points = tf.transpose(tf.gather(tf.transpose(points), self.active_dims)) #tf 1.1 gather on axis=0 only

    return points

