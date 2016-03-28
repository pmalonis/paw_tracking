from __future__ import print_function

__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T


class Localization(object):
    """Localization regression class"""

    def __init__(self, input, n_in, n_out=2):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.y_pred = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def L2_error(self, y):
        return ((self.y_pred - y)**2).sum()
        
    def mean_errors(self, y):
        """
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('float'):
            try:
                print(self.y_pred.eval(), y.eval())
            except:
                pass
            e = T.mean(((self.y_pred - y)**2).sum(axis=1))
            return e
        else:
            raise NotImplementedError()
