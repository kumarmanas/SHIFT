import numpy as np
import six
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import transpose as transpose_lib
from tensorflow_probability.python.distributions import variational_gaussian_process as variational_gaussian_process_lib
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import transformed_distribution as transformed_distribution_lib
from tensorflow_probability.python.distributions import normal as normal_lib
tfd = tfp.distributions

from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
import edward2 as ed2

def _deserialize(name, custom_objects=None):
  return tf.keras.utils.legacy.deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='convert-to-tensor function',
  )


def _get_convert_to_tensor_fn(identifier):
  """Return a convert-to-tensor func, given a name, config, callable, etc."""
  if identifier is None:
    return None

  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return _deserialize(identifier)

  if isinstance(identifier, dict):
    return _deserialize(identifier)

  if isinstance(identifier, property):
    identifier = identifier.fget
  if callable(identifier):
    return identifier

  raise ValueError('Could not interpret '
                   'convert-to-tensor function identifier:', identifier)


def mapping_wrapper(mapper_func):
    def inner(y_true, y_pred, **kwargs):
        #y_pred = y_pred.mean() # TODO: replace with conversion to tensor specified earlier
        mapper_func(y_true, y_pred, **kwargs)
    return inner

def convert_to_tensor_activation(tfd_rv, activation):
    logits = ed2.layers.utils.mean_field_logits(tfd_rv.mean(), tfd_rv.variance(), mean_field_factor=np.pi/8)
    if activation == "softmax":
        return tf.nn.softmax(logits)
    elif activation == "sigmoid":
        return tf.nn.sigmoid(logits)

class DUELogits(tf.keras.layers.Layer):
    def __init__(self, activation, mean_field_factor=np.pi/3, dtype="float32", **kwargs):
        self.mean_field_factor = mean_field_factor
        self.activation = activation
        self.type = dtype
        super().__init__(**kwargs)

    def call(self, logits, training=None):
        # if not training:
        logits = ed2.layers.utils.mean_field_logits(logits.mean(), logits.variance(), mean_field_factor=self.mean_field_factor)
        logits = tf.keras.layers.Activation(self.activation, dtype=self.type)(logits)
        return logits

class RBFKernelFn(tf.keras.layers.Layer):
  def __init__(self, amplitude, length_scale, **kwargs):
    super(RBFKernelFn, self).__init__(**kwargs)
    dtype = kwargs.get('dtype', None)

    self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(amplitude),
            dtype=dtype,
            name='amplitude')

    self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(length_scale),
            dtype=dtype,
            name='length_scale')

  def call(self, x):
    # Never called -- this is just a layer so it can hold variables
    # in a way Keras understands.
    return x

  @property
  def kernel(self):
    return tfp.math.psd_kernels.ExponentiatedQuadratic(
      amplitude=tf.nn.softplus(self._amplitude),
      length_scale=tf.nn.softplus(self._length_scale)
    )

