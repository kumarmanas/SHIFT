"""
ResNet50 model with heteroscedastic Spectral Normalization

"""

from uncertainty_baselines.models.resnet50_hetsngp import (
  MonteCarloDropout,
  make_random_feature_initializer,
  resnet50_hetsngp,
  resnet50_hetsngp_add_last_layer
)

import functools
import string
from keras.utils import control_flow_util
from tensorflow.keras import backend
import numpy as np

import edward2 as ed2
import tensorflow as tf

class SpectralNormalization(ed2.layers.SpectralNormalization):

    def build(self, input_shape):
        super(ed2.layers.SpectralNormalization, self).build(input_shape)
        self.layer.kernel._aggregation = self.aggregation  # pylint: disable=protected-access
        self._dtype = self.layer.kernel.dtype

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.uv_initializer = tf.initializers.random_normal()

        self.v = self.add_weight(
            shape=(1, np.prod(self.w_shape[:-1])),
            initializer=self.uv_initializer,
            trainable=False,
            name='v',
            dtype=self.dtype,
            aggregation=self.aggregation)

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=self.uv_initializer,
            trainable=False,
            name='u',
            dtype=self.dtype,
            aggregation=self.aggregation)
        
        self.r = self.add_weight(
            shape=self.w.shape,
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
            name="r",
            dtype=self.dtype
        )

        self.update_weights()

class SpectralNormalizationConv2D(ed2.layers.SpectralNormalizationConv2D):

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.layer.kernel._aggregation = self.aggregation  # pylint: disable=protected-access
        self._dtype = self.layer.kernel.dtype

        # Shape (kernel_size_1, kernel_size_2, in_channel, out_channel).
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.strides = self.layer.strides

        # Set the dimensions of u and v vectors.
        batch_size = input_shape[0]
        uv_dim = batch_size if self.legacy_mode else 1

        # Resolve shapes.
        in_height = input_shape[1]
        in_width = input_shape[2]
        in_channel = self.w_shape[2]

        out_height = in_height // self.strides[0]
        out_width = in_width // self.strides[1]
        out_channel = self.w_shape[3]

        self.in_shape = (uv_dim, in_height, in_width, in_channel)
        self.out_shape = (uv_dim, out_height, out_width, out_channel)
        self.uv_initializer = tf.initializers.random_normal()

        self.v = self.add_weight(
            shape=self.in_shape,
            initializer=self.uv_initializer,
            trainable=False,
            name='v',
            dtype=self.dtype,
            aggregation=self.aggregation)

        self.u = self.add_weight(
            shape=self.out_shape,
            initializer=self.uv_initializer,
            trainable=False,
            name='u',
            dtype=self.dtype,
            aggregation=self.aggregation)
        
        self.r = self.add_weight(
            shape=self.w.shape,
            initializer=tf.keras.initializers.Ones(),
            trainable=False,
            name="r",
            dtype=self.dtype
        )
        # print(f"Input shape: {input_shape}")
        # print(f"W shape: {self.w_shape}")
        # print(f"In shape: {self.in_shape}")
        # print(f"Out shape: {self.out_shape}")
        super(ed2.layers.SpectralNormalizationConv2D, self).build()



# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

def make_conv2d_layer(use_spec_norm,
                      spec_norm_iteration,
                      spec_norm_bound):
    """Defines type of Conv2D layer to use based on spectral normalization."""
    Conv2DBase = functools.partial(tf.keras.layers.Conv2D, padding='same')  # pylint: disable=invalid-name
    def Conv2DNormed(*conv_args, **conv_kwargs):  # pylint: disable=invalid-name
        return SpectralNormalizationConv2D(
            Conv2DBase(*conv_args, **conv_kwargs),
            iteration=spec_norm_iteration,
            norm_multiplier=spec_norm_bound)

    return Conv2DNormed if use_spec_norm else Conv2DBase


def make_batchnorm_layer(use_spec_norm,
                         spec_norm_iteration,
                         spec_norm_bound):
    BatchNormBase = functools.partial(tf.keras.layers.BatchNormalization, 
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
        )
    BatchNormNormed = functools.partial(SpectralNormalizedBatchNormalization,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            norm_multiplier=spec_norm_bound,
        )

    return BatchNormNormed if use_spec_norm else BatchNormBase

def bottleneck_block(inputs, filters, stage, block, strides, conv_layer, batchnorm_layer,
                     dropout_layer):
  """Residual block with 1x1 -> 3x3 -> 1x1 convs in main path.

  Note that strides appear in the second conv (3x3) rather than the first (1x1).
  This is also known as "ResNet v1.5" as it differs from He et al. (2015)
  (http://torch.ch/blog/2016/02/04/resnets.html).

  Args:
    inputs: tf.Tensor.
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    conv_layer: tf.keras.layers.Layer.
    dropout_layer: Callable for dropout layer.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = conv_layer(
      filters1,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2a')(
          inputs)
  x = batchnorm_layer(name=bn_name_base + '2a')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = dropout_layer(x)

  x = conv_layer(
      filters2,
      kernel_size=3,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2b')(
          x)
  x = batchnorm_layer(name=bn_name_base + '2b')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = dropout_layer(x)

  x = conv_layer(
      filters3,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2c')(
          x)
  x = batchnorm_layer(name=bn_name_base + '2c')(x)

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = conv_layer(
        filters3,
        kernel_size=1,
        use_bias=False,
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '1')(
            shortcut)
    shortcut = batchnorm_layer(name=bn_name_base + '1')(shortcut)
    shortcut = dropout_layer(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, num_blocks, stage, strides, conv_layer, batchnorm_layer,
          dropout_layer):
  """Group of residual blocks."""
  blocks = string.ascii_lowercase
  x = bottleneck_block(
      inputs,
      filters,
      stage,
      block=blocks[0],
      strides=strides,
      conv_layer=conv_layer,
      batchnorm_layer=batchnorm_layer,
      dropout_layer=dropout_layer)
  for i in range(num_blocks - 1):
    x = bottleneck_block(
        x,
        filters,
        stage,
        block=blocks[i + 1],
        strides=1,
        conv_layer=conv_layer,
        batchnorm_layer=batchnorm_layer,
        dropout_layer=dropout_layer)
  return x

def resnet50_hetsn(input_shape,
                     batch_size,
                     num_classes,
                     num_factors,
                     use_mc_dropout,
                     dropout_rate,
                     filterwise_dropout,
                     use_gp_layer,
                     gp_hidden_dim,
                     gp_scale,
                     gp_bias,
                     gp_input_normalization,
                     gp_random_feature_type,
                     gp_cov_discount_factor,
                     gp_cov_ridge_penalty,
                     gp_output_imagenet_initializer,
                     use_spec_norm,
                     spec_norm_iteration,
                     spec_norm_bound,
                     temperature,
                     num_mc_samples=100,
                     eps=1e-5,
                     sngp_var_weight=1.,
                     het_var_weight=1.,
                     omit_last_layer=False):
    """Builds ResNet50.

    Using strided conv, pooling, four groups of residual blocks, and pooling, the
    network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
    14x14 -> 7x7 (Table 1 of He et al. (2015)).

    Args:
    input_shape: Shape tuple of input excluding batch dimension.
    batch_size: The batch size of the input layer. Required by the spectral
        normalization.
    num_classes: Number of output classes.
    use_mc_dropout: Whether to apply Monte Carlo dropout.
    dropout_rate: Dropout rate.
    filterwise_dropout:  Dropout whole convolutional filters instead of
        individual values in the feature map.
    use_gp_layer: Whether to use Gaussian process layer as the output layer.
    gp_hidden_dim: The hidden dimension of the GP layer, which corresponds to
        the number of random features used for the approximation.
    gp_scale: The length-scale parameter for the RBF kernel of the GP layer.
    gp_bias: The bias term for GP layer.
    gp_input_normalization: Whether to normalize the input using LayerNorm for
        GP layer. This is similar to automatic relevance determination (ARD) in
        the classic GP learning.
    gp_random_feature_type: The type of random feature to use for
        `RandomFeatureGaussianProcess`.
    gp_cov_discount_factor: The discount factor to compute the moving average of
        precision matrix.
    gp_cov_ridge_penalty: Ridge penalty parameter for GP posterior covariance.
    gp_output_imagenet_initializer: Whether to initialize GP output layer using
        Gaussian with small standard deviation (sd=0.01).
    use_spec_norm: Whether to apply spectral normalization.
    spec_norm_iteration: Number of power iterations to perform for estimating
        the spectral norm of weight matrices.
    spec_norm_bound: Upper bound to spectral norm of weight matrices.
    omit_last_layer: Optional. Omits the last pooling layer if it is set to
        True.

    Returns:
    tf.keras.Model.
    """
    dropout_layer = functools.partial(
        MonteCarloDropout,
        dropout_rate=dropout_rate,
        use_mc_dropout=use_mc_dropout,
        filterwise_dropout=filterwise_dropout)
    conv_layer = make_conv2d_layer(use_spec_norm=use_spec_norm,
                                    spec_norm_iteration=spec_norm_iteration,
                                    spec_norm_bound=spec_norm_bound)
    batchnorm_layer = make_batchnorm_layer(use_spec_norm=use_spec_norm,
                                    spec_norm_iteration=spec_norm_iteration,
                                    spec_norm_bound=spec_norm_bound)

    inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
    x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)
    if use_spec_norm: # apply Spectral Normalization to input layer as well (uses different padding)
        x = SpectralNormalizationConv2D(
                tf.keras.layers.Conv2D(
                    64,
                    kernel_size=7,
                    strides=2,
                    padding='valid',
                    use_bias=False,
                    kernel_initializer='he_normal',
                    name='conv1'),
                iteration=spec_norm_iteration,
                norm_multiplier=spec_norm_bound)(x)
    else:
       x = tf.keras.layers.Conv2D(
                    64,
                    kernel_size=7,
                    strides=2,
                    padding='valid',
                    use_bias=False,
                    kernel_initializer='he_normal',
                    name='conv1')(x)

    x = batchnorm_layer(name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = dropout_layer(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    x = group(
        x, [64, 64, 256],
        stage=2,
        num_blocks=3,
        strides=1,
        conv_layer=conv_layer,
        batchnorm_layer=batchnorm_layer,
        dropout_layer=dropout_layer)
    x = group(
        x, [128, 128, 512],
        stage=3,
        num_blocks=4,
        strides=2,
        conv_layer=conv_layer,
        batchnorm_layer=batchnorm_layer,
        dropout_layer=dropout_layer)
    x = group(
        x, [256, 256, 1024],
        stage=4,
        num_blocks=6,
        strides=2,
        conv_layer=conv_layer,
        batchnorm_layer=batchnorm_layer,
        dropout_layer=dropout_layer)
    x = group(
        x, [512, 512, 2048],
        stage=5,
        num_blocks=3,
        strides=2,
        conv_layer=conv_layer,
        batchnorm_layer=batchnorm_layer,
        dropout_layer=dropout_layer)

    if omit_last_layer:
        return tf.keras.Model(inputs=inputs, outputs=x, name='resnet50')

    return resnet50_hetsngp_add_last_layer(
        inputs, x, num_classes, num_factors, use_gp_layer, gp_hidden_dim,
        gp_scale, gp_bias, gp_input_normalization, gp_random_feature_type,
        gp_cov_discount_factor, gp_cov_ridge_penalty,
        gp_output_imagenet_initializer, temperature, num_mc_samples, eps,
        sngp_var_weight, het_var_weight)


class SpectralNormalizedBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, norm_multiplier, *args, **kwargs):
        self.norm_multiplier = norm_multiplier
        super(SpectralNormalizedBatchNormalization, self).__init__(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        training = self._get_training_value(training)

        if self.virtual_batch_size is not None:
            # Virtual batches (aka ghost batches) can be simulated by reshaping the
            # Tensor and reusing the existing batch norm implementation
            original_shape = tf.shape(inputs)
            original_shape = tf.concat(
                [tf.constant([-1]), original_shape[1:]], axis=0)
            expanded_shape = tf.concat([
                tf.constant([self.virtual_batch_size, -1]),
                original_shape[1:]
            ], axis=0)

            # Will cause errors if virtual_batch_size does not divide the batch size
            inputs = tf.reshape(inputs, expanded_shape)

            def undo_virtual_batching(outputs):
                outputs = tf.reshape(outputs, original_shape)
                return outputs

        if self.fused:
            outputs = self._fused_batch_norm(inputs, training=training)
            if self.virtual_batch_size is not None:
                # Currently never reaches here since fused_batch_norm does not support
                # virtual batching
                outputs = undo_virtual_batching(outputs)
            return outputs

        inputs_dtype = inputs.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            # Do all math in float32 if given 16-bit inputs for numeric stability.
            # In particular, it's very easy for variance to overflow in float16 and
            # for safety we also choose to cast bfloat16 to float32.
            inputs = tf.cast(inputs, tf.float32)

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        if self.virtual_batch_size is not None:
            del reduction_axes[1]  # Do not reduce along virtual batch dim

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and
                reduction_axes != list(range(ndims - 1))):
                return tf.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = control_flow_util.constant_value(training)
        if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
            mean, variance = self.moving_mean, self.moving_variance
        else:
            if self.adjustment:
                adj_scale, adj_bias = self.adjustment(tf.shape(inputs))
                # Adjust only during training.
                adj_scale = control_flow_util.smart_cond(
                    training, lambda: adj_scale, lambda: tf.ones_like(adj_scale))
                adj_bias = control_flow_util.smart_cond(
                    training, lambda: adj_bias, lambda: tf.zeros_like(adj_bias))
                scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

        # Some of the computations here are not necessary when training==False
        # but not a constant. However, this makes the code simpler.
        keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
        mean, variance = self._moments(
            tf.cast(inputs, self._param_dtype),
            reduction_axes,
            keep_dims=keep_dims)

        moving_mean = self.moving_mean
        moving_variance = self.moving_variance

        mean = control_flow_util.smart_cond(
            training, lambda: mean,
            lambda: tf.convert_to_tensor(moving_mean))
        variance = control_flow_util.smart_cond(
            training, lambda: variance,
            lambda: tf.convert_to_tensor(moving_variance))

        if self.virtual_batch_size is not None:
            new_mean = tf.reduce_mean(mean, axis=1, keepdims=True)
            new_variance = tf.reduce_mean(variance, axis=1, keepdims=True)
        else:
            new_mean, new_variance = mean, variance

            if self._support_zero_size_input():
                # Keras assumes that batch dimension is the first dimension for Batch
                # Normalization.
                input_batch_size = tf.shape(inputs)[0]
            else:
                input_batch_size = None

            if self.renorm:
                r, d, new_mean, new_variance = self._renorm_correction_and_moments(
                    new_mean, new_variance, training, input_batch_size)
                # When training, the normalized values (say, x) will be transformed as
                # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
                # = x * (r * gamma) + (d * gamma + beta) with renorm.
                r = _broadcast(tf.stop_gradient(r, name='renorm_r'))
                d = _broadcast(tf.stop_gradient(d, name='renorm_d'))
                scale, offset = _compose_transforms(r, d, scale, offset)

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(var, value, self.momentum,
                                                input_batch_size)

            def mean_update():
                true_branch = lambda: _do_update(self.moving_mean, new_mean)
                false_branch = lambda: self.moving_mean
                return control_flow_util.smart_cond(training, true_branch, false_branch)

            def variance_update():
                """Update the moving variance."""

                def true_branch_renorm():
                    # We apply epsilon as part of the moving_stddev to mirror the training
                    # code path.
                    moving_stddev = _do_update(self.moving_stddev,
                                                tf.sqrt(new_variance + self.epsilon))
                    return self._assign_new_value(
                        self.moving_variance,
                        # Apply relu in case floating point rounding causes it to go
                        # negative.
                        backend.relu(moving_stddev * moving_stddev - self.epsilon))

                if self.renorm:
                    true_branch = true_branch_renorm
                else:
                    true_branch = lambda: _do_update(self.moving_variance, new_variance)

                    false_branch = lambda: self.moving_variance
                return control_flow_util.smart_cond(training, true_branch, false_branch)

            self.add_update(mean_update)
            self.add_update(variance_update)

        # spectral normalize scale coefficients
        lipschitz = tf.math.reduce_max(tf.abs(scale*variance+self.epsilon)**-0.05)
        lipschitz_factor = tf.math.maximum(lipschitz/self.norm_multiplier, tf.ones(lipschitz.shape))
        scale = scale/lipschitz_factor

        mean = tf.cast(mean, inputs.dtype)
        variance = tf.cast(variance, inputs.dtype)
        if offset is not None:
            offset = tf.cast(offset, inputs.dtype)
        if scale is not None:
            scale = tf.cast(scale, inputs.dtype)
        outputs = tf.nn.batch_normalization(
            inputs,
            _broadcast(mean),
            _broadcast(variance),
            offset,
            scale,
            self.epsilon,
        )
        if inputs_dtype in (tf.float16, tf.bfloat16):
            outputs = tf.cast(outputs, inputs_dtype)

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        if self.virtual_batch_size is not None:
            outputs = undo_virtual_batching(outputs)
        return outputs
    
    def get_config(self):
        config = {"norm_multiplier": self.norm_multiplier}
        base_config = super().get_config()
        return dict(list(base_config.items())+list(config.items()))
