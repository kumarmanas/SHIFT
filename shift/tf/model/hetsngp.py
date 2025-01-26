_SUPPORTED_LIKELIHOOD = ('binary_logistic', 'poisson', 'gaussian') # only uses gaussian to approximate
#Main function to introduce heteroscedasticity in the model
import edward2 as ed2
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter

tfd = tfp.distributions


class SNGPResetCovariance(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        """
        Resets covariance matrix of SNGP model at the beginning of epoch
        """
        if epoch > 0:
            tf.print("Resetting Covariance Matrix")
            self.model.output_layer.reset_covariance_matrix()

class LogitsAdjuster(tf.keras.layers.Layer):
    def __init__(self, mc_samples=0, mean_field_factor=1, **kwargs):
        self.mc_samples = mc_samples
        self.mean_field_factor = mean_field_factor
        super().__init__(**kwargs)

    def call(self, logits, covariances, training=None):
        if not training:
            if self.mc_samples > 0:
                # generate posterior, average over samples drawn from it
                bnd = tfd.MultivariateNormalDiag(loc=tf.transpose(logits), scale_diag=tf.linalg.diag_part(covariances))
                logits = tf.transpose(tf.reduce_mean(bnd.sample(self.mc_samples),axis=0))
                #logits = tf.transpose(bnd.sample(self.mc_samples))
            else:
                logits = ed2.layers.utils.mean_field_logits(logits, covariances, mean_field_factor=self.mean_field_factor)
        return logits

class ModeRegularizer(tf.keras.regularizers.Regularizer):
    """
    Recovers l2 regularizer for zero mean isotropic Gaussian prior
    """
    def __init__(self, weight, mean=None, precision=None, diagonal=False):
        self.weight = weight
        self.mean = mean
        self.precision = precision
        self.diagonal = diagonal
    
    def __call__(self, x):
        if self.mean is None:
            penalty = tf.reduce_sum(self.weight/2*tf.square(x))
        elif self.precision is None:
            penalty = tf.reduce_sum(self.weight/2*tf.square(x-self.mean))
        elif self.diagonal:
            penalty = tf.reduce_sum(self.weight/2*tf.matmul(tf.expand_dims(tf.linalg.tensor_diag_part(self.precision),0),tf.square(x-self.mean)))
        else:
            penalty = tf.reduce_sum(self.weight/2*tf.linalg.tensor_diag_part(tf.matmul(tf.transpose(tf.square(x-self.mean)),tf.matmul(self.precision,tf.square(x-self.mean)))))
        return penalty
    
    def get_config(self):
        config = {"weight": self.weight, "mean": self.mean, "precision": self.precision}
        return config
    
class RandomFeatureGaussianProcess(ed2.layers.RandomFeatureGaussianProcess):
    """Gaussian process layer with random feature approximation.

    During training, the model updates the maximum a posteriori (MAP) logits
    estimates and posterior precision matrix using minibatch statistics. During
    inference, the model divides the MAP logit estimates by the predictive
    standard deviation, which is equivalent to approximating the posterior mean
    of the predictive probability via the mean-field approximation.

    User can specify different types of random features by setting
    `use_custom_random_features=True`, and change the initializer and activations
    of the custom random features. For example:

    MLP Kernel: initializer='random_normal', activation=tf.nn.relu
    RBF Kernel: initializer='random_normal', activation=tf.math.cos

    A linear kernel can also be specified by setting gp_kernel_type='linear' and
    `use_custom_random_features=True`.

    Attributes:
    units: (int) The dimensionality of layer.
    num_inducing: (int) The number of random features for the approximation.
    is_training: (tf.bool) Whether the layer is set in training mode. If so the
        layer updates the Gaussian process' variance estimate using statistics
        computed from the incoming minibatches.
    """   
    def __init__(self,
                dataset_size,
                **kwargs):
        super(RandomFeatureGaussianProcess, self).__init__(**kwargs)
        self.dataset_size = dataset_size

    def _build_sublayer_classes(self):
        """Defines sublayer classes."""
        self.bias_layer = tf.Variable
        self.dense_layer = tf.keras.layers.Dense
        self.covariance_layer = LaplaceRandomFeatureCovariance
        self.input_normalization_layer = tf.keras.layers.LayerNormalization

    def build(self, input_shape):
        self._build_sublayer_classes()
        if self.normalize_input:
            self._input_norm_layer = self.input_normalization_layer(
                name='gp_input_normalization')
        self._input_norm_layer.build(input_shape)
        input_shape = self._input_norm_layer.compute_output_shape(input_shape)

        self._random_feature = self._make_random_feature_layer(
            name='gp_random_feature')
        self._random_feature.build(input_shape)
        input_shape = self._random_feature.compute_output_shape(input_shape)

        if self.return_gp_cov:
            self._gp_cov_layer = self.covariance_layer(
                momentum=self.gp_cov_momentum,
                ridge_penalty=self.gp_cov_ridge_penalty,
                likelihood=self.gp_cov_likelihood,
                dataset_size=self.dataset_size,
                dtype=self.dtype,
                name='gp_covariance')
        self._gp_cov_layer.build(input_shape)

        # zero-mean isotropic Gaussian prior equivalent to l2 regularizer
        kernel_regularizer = ModeRegularizer( 
            weight=self.l2_regularization,
            mean = tf.zeros((self.num_inducing,self.units), dtype=self.dtype),
            precision = tf.eye(self.num_inducing, dtype=self.dtype)
            )
        self._gp_output_layer = self.dense_layer(
            units=self.units,
            use_bias=False,
            kernel_regularizer=kernel_regularizer, #tf.keras.regularizers.l2(self.l2_regularization),
            dtype=self.dtype,
            name='gp_output_weights',
            **self.gp_output_kwargs)
        self._gp_output_layer.build(input_shape)

        self._gp_output_bias = self.bias_layer(
            initial_value=[self.gp_output_bias] * self.units,
            dtype=self.dtype,
            trainable=self.gp_output_bias_trainable,
            name='gp_output_bias')

        self.built = True


class LaplaceRandomFeatureCovariance(ed2.layers.LaplaceRandomFeatureCovariance):
    def __init__(self,
                dataset_size,
                **kwargs):
        super(LaplaceRandomFeatureCovariance, self).__init__(**kwargs)
        self.dataset_size = dataset_size

    def make_precision_matrix_update_op(self,
                                      gp_feature,
                                      logits,
                                      precision_matrix):
        if self.likelihood != 'gaussian':
            if logits is None:
                raise ValueError(
                    f'"logits" cannot be None when likelihood={self.likelihood}')

            if logits.shape[-1] != 1:
                raise ValueError(
                    f'likelihood={self.likelihood} only support univariate logits.'
                    f'Got logits dimension: {logits.shape[-1]}')

        batch_size = tf.shape(gp_feature)[0]
        batch_size = tf.cast(batch_size, dtype=gp_feature.dtype)

        # Computes batch-specific normalized precision matrix.
        if self.likelihood == 'binary_logistic':
            prob = tf.sigmoid(logits)
            prob_multiplier = prob * (1. - prob)
        elif self.likelihood == 'poisson':
            prob_multiplier = tf.exp(logits)
        else:
            prob_multiplier = 1.

        gp_feature_adjusted = tf.sqrt(prob_multiplier) * gp_feature
        precision_matrix_minibatch = tf.matmul(
            gp_feature_adjusted, gp_feature_adjusted, transpose_a=True)

        # Updates the population-wise precision matrix.
        if self.momentum > 0: # TODO: add temperature
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                self.momentum * precision_matrix +
                (1. - self.momentum) * precision_matrix_minibatch)
        else:
        # Compute exact population-wise covariance without momentum.
        # If use this option, make sure to pass through data only once.
            precision_matrix_new = precision_matrix + precision_matrix_minibatch

        # Returns the update op.
        return precision_matrix.assign(precision_matrix_new)

#from ed2.tensorflow.layers import RandomFeatureGaussianProcess
#from ed2.tensorflow.layers import MCSoftmaxDenseFA

class HeteroscedasticSNGPLayer(ed2.layers.HeteroscedasticSNGPLayer):
    def __init__(self,
                 dataset_size,
                 num_classes,
                 num_factors=10,
                 temperature=1.0,
                 train_mc_samples=1000,
                 test_mc_samples=1000,
                 compute_pred_variance=True,
                 share_samples_across_batch=False,
                 logits_only=False,
                 sngp_var_weight=1.0,  # Add this parameter
                 het_var_weight=1.0,   # Add this parameter
                 **kwargs):
        super(HeteroscedasticSNGPLayer, self).__init__(
            num_classes=num_classes,
            num_factors=num_factors,
            temperature=temperature,
            train_mc_samples=train_mc_samples,
            test_mc_samples=test_mc_samples,
            compute_pred_variance=compute_pred_variance,
            share_samples_across_batch=share_samples_across_batch,
            logits_only=logits_only,
            **kwargs
        )
        self.dataset_size = dataset_size
        self.sngp_var_weight = sngp_var_weight  # Store the parameter
        self.het_var_weight = het_var_weight    # Store the parameter

        self.sngp_layer = RandomFeatureGaussianProcess(
            units=num_classes,
            dataset_size=dataset_size,
            **kwargs
        )

        self.het_layer = ed2.layers.MCSoftmaxDenseFA(
            num_classes=num_classes,
            num_factors=num_factors,
            temperature=temperature,
            train_mc_samples=train_mc_samples,
            test_mc_samples=test_mc_samples,
            compute_pred_variance=compute_pred_variance,
            share_samples_across_batch=share_samples_across_batch,
            logits_only=logits_only
        )

    def call(self, inputs, training=None):
        sngp_logits, sngp_covmat = self.sngp_layer(inputs)
        het_outputs = self.het_layer(sngp_logits)

        if self.logits_only:
            return het_outputs

        logits, log_probs, probs, pred_variance = het_outputs

        if self.compute_pred_variance:
            # Combine SNGP and heteroscedastic variances
            sngp_var = tf.linalg.diag_part(sngp_covmat) * self.sngp_var_weight
            combined_var = pred_variance * self.het_var_weight + sngp_var

            return logits, log_probs, probs, combined_var
        else:
            return logits, log_probs, probs, None

    def reset_covariance_matrix(self):
        self.sngp_layer.reset_covariance_matrix()

