"""Linear module."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function
import numpy as np

import sonnet as snt
import tensorflow as tf


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class MyGCNLayer(tf.keras.layers.Layer):
    def __init__(self, name="MyGCNLayer", outputdim=None, act=True):
        super(MyGCNLayer, self).__init__()
        self.num_outputs = outputdim
        self.act = act

    def build(self, input_shape):
        input_size = input_shape[1]
        self.kernel = self.add_weight("kernel", shape=[input_size, self.num_outputs])



    def call(self, input,a_hat):

        x = tf.nn.dropout(input, 1 - 0.5)
        x = tf.matmul(x, self.kernel)
        x = tf.sparse.sparse_dense_matmul(a_hat, x)
        if self.act:
            return tf.nn.relu(x)
        else:
            return x


class MLPGraphNetwork(snt.Module):
    def __init__(self, name="MLPGraphNetwork", outputdim=None, act=tf.nn.relu, bias=True):
        """

    :type outputdim: object
    """
        super(MLPGraphNetwork, self).__init__(name=name)
        self.output_dim = outputdim
        self.bias = bias
        self.act = act

    @snt.once
    def _initialize(self, inputs):
        input_size = inputs.shape[1]
        self.w = glorot([input_size, self.output_dim], name="weights")
        if self.bias:
            # Fix this, the shape of the bias is not automatized .  it was giving me an error

            self.b_arr = tf.Variable(tf.zeros((2708, 1), dtype=tf.float32), name="bias")
            print("Bias done", self.b_arr)
        # tf.Variable(tf.random.normal([input_size, self.output_size]))

    def __call__(self, inputs, a_hat):
        self._initialize(inputs)
        x = tf.nn.dropout(inputs, 1 - 0.5)
        res = tf.matmul(x, self.w)
        output = tf.matmul(a_hat, res)
        if self.bias:
            output += self.b_arr

        return self.act(output)


class GCN(tf.keras.Model):

    def __init__(self,
                 encoder_arr=None,
                 pi=None,
                 convolution_kernel_1=None,
                 convolution_kernel_2=None,
                 decoder_arr=None,
                 name="GCN"):
        super(GCN, self).__init__(name=name)
        # self._normalizer = snt.LayerNorm(axis=1, create_offset=True, create_scale=True)
        self._encoder = snt.Sequential([snt.nets.MLP(encoder_arr, activate_final=True),snt.LayerNorm(axis=1, create_offset=True, create_scale=True)])
        #self._encoder = snt.LayerNorm(axis=0, create_offset=True, create_scale=True)
        self._graphNetwork = MyGCNLayer(outputdim=convolution_kernel_1, name="gcn1",act=True)
        self._conv2 = MyGCNLayer(outputdim=convolution_kernel_2, name="gcn2",act=True)
        # self._decoder = snt.Sequential([snt.LayerNorm(axis=1, create_offset=True, create_scale=True),snt.nets.MLP(decoder_arr, activate_final=False)])

    def call(self, input_op, dadmx):
        x=self._encoder(input_op)
        conv1 = self._graphNetwork(x, dadmx)
        conv2 = self._conv2(conv1,dadmx)
        return conv2
