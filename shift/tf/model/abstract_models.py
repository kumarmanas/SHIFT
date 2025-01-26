'''
Abstract base class and mixins
'''
from abc import ABC, abstractmethod
from functools import partialmethod, partial
import logging

#from shift.tf.model.resnet50_sn import SpectralNormalization
from shift.tf.model.resnet50_hetsn import SpectralNormalization
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm
from uncertainty_baselines.models.resnet50_hetsngp import (
    make_random_feature_initializer,
)
from uncertainty_baselines.models.variational_utils import init_kernel_regularizer
from tensorflow.python.keras.engine import data_adapter
from shift.tf.util import MultivariateNormalDiag

from shift.tf.model.due import (
    RBFKernelFn,
    #VariationalGaussianProcess,
    DUELogits,
    convert_to_tensor_activation
)
from shift.tf.model.hetsngp import (
    RandomFeatureGaussianProcess, 
    ModeRegularizer, 
    LogitsAdjuster,
    HeteroscedasticSNGPLayer,
)
import edward2 as ed2
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow_probability.python.distributions import kullback_leibler, Normal
from tensorflow_probability.python.distributions.kullback_leibler import _DIVERGENCES
from typing import Tuple
from scipy.cluster.vq import kmeans2


class ModelFactory(ABC):
    """
    Model factory for any predictor

    :param int dataset_size: Number of examples in the dataset
    :param dict **kwargs: Hyperparameters
    """

    def __init__(self, dataset_size, **kwargs):
        self.dataset_size = dataset_size
        self.dataset = kwargs.pop("dataset", None)
        self.hyperparameters = kwargs
        self.logger = logging.getLogger('trainer')
        self.scheduler = None

    def create_model(self, inputs, outputs, optimizer):
        """
        Instanciates the keras model

        :param list inputs: list of input layers
        :param list outputs: list of output layers
        :param optimizer: tensorflow optimizer
        :return: the model
        :rtype: Model
        """
        outputs = {
            k: layers.Activation('linear', dtype='float32', name=k)(v)
            for k, v in outputs.items()
        }
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_loss(self):
        """
        Must return a loss function

        :return: Loss function
        :rtype: Callable
        """
        raise NotImplementedError

    def get_mapper(self):
        """
        Has to return functions mapping y_true, y_pred into coordinate (trajectory) space and categorical space.

        :return: mapper to categorical, mappter to multi-categorical, mapper to trajectory
        :rtype: Callable, Callable, Callable
        """
        raise NotImplementedError

    def get_backbone(self, shape):
        """
        Must build the feature projector backbone

        :param tuple shape: Input shape
        :return: inputs, output
        :rtype: list[tf.Tensor], tf.Tensor
        """
        raise NotImplementedError

    def get_hidden(self, logits, hidden_layer_size):
        """
        Builds a single, hidden layer

        :param tf.Tensor logits: Output from the last layer
        :param int hidden_layer_size: Number of hidden units
        :return: layer output
        :rtype: tf.Tensor
        """
        raise NotImplementedError

    def get_head(self, logits, num_modes, activation):
        """
        Builds a prediction output layer

        :param tf.Tensor logits: Output from the last layer
        :param int num_modes: Number of output values
        :param bool is_classifier: True if classifier, False for regressor
        :param bool is_multilabel: True for multilabel classification, False for single class prediction
        :return: layer output
        :rtype: tf.Tensor
        """
        raise NotImplementedError

    def build(self):
        """
        Builds the model itself

        :return: model, loss, optimizer, scheduler, prediction to trajectory mapper, groundtruth to class mapper, groundtruth to label mapper
        :rtype: Model, func, Optimizer, Schedule, func, func, func
        """
        inputs, output = self.get_backbone((480, 480, 3))

        for hidden_layer_size in [4096]:
            logits = self.get_hidden(output, hidden_layer_size)

        y_pred = {}
        
        
        for key, definition in self.heads.items():
            activation, shape, reuse_head = definition
            if activation is None:
                activation = "linear"
            if isinstance(reuse_head, str):
                y_pred[key] = layers.Activation(activation, dtype='float32')(y_pred[reuse_head]._keras_history[0].input)
            else:
                y_pred[key] = self.get_head(logits, shape, activation)

        optimizer = SGD(
            learning_rate=self.hyperparameters["lr"]
            if self.scheduler is None
            else self.scheduler,
            momentum=self.hyperparameters['momentum'],
        )
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        model = self.create_model(
            inputs=inputs,
            outputs=y_pred,
            optimizer=optimizer,
        )
        model.feature_extractor = Model(inputs=inputs, outputs=logits)
        loss = self.get_loss()

        class_mapper, label_mapper, traj_mapper = self.get_mapper(model)

        return (model, loss, optimizer, traj_mapper, class_mapper, label_mapper)

class SNGPMixin():
    pass
    """
    Mixin for Spectral Normalized Gaussian Process outputs
    """
    # def __init__(self):
    #     if self.hyperparameters['gp_posterior_temp'] is None:
    #         self.hyperparameters['gp_posterior_temp'] = (
    #             self.hyperparameters['batch_size']
    #         )
    #     if self.hyperparameters['extractor_posterior_temp'] is None:
    #         self.hyperparameters['extractor_posterior_temp'] = (
    #             self.hyperparameters['batch_size']
    #         )

    #     self.effective_dataset_size = (
    #         (self.dataset_size) /
    #         self.hyperparameters['gp_posterior_temp']
    #     )


    #     self.extractor_effective_dataset_size = (
    #         (self.dataset_size) /
    #         self.hyperparameters['extractor_posterior_temp']
    #     )

    # def get_hidden(self, logits, hidden_layer_size):

    #     if self.hyperparameters["use_spec_norm"]:
    #         logits = SpectralNormalization(
    #             layers.Dense(hidden_layer_size, activation="relu", use_bias=False),
    #             norm_multiplier=self.hyperparameters["spec_norm_bound"],
    #         )(logits)
    #     else:
    #         logits = layers.Dense(hidden_layer_size, activation="relu", use_bias=False)(logits)
    #     return logits

    # def get_head(self, logits, num_modes, activation):

    #     if self.hyperparameters["use_gp_layer"]:
    #         gp_output_initializer = gp_output_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)

    #         self.output_layer = RandomFeatureGaussianProcess(
    #             units=num_modes,
    #             num_inducing=self.hyperparameters["gp_num_inducing"],
    #             gp_kernel_scale=self.hyperparameters["gp_kernel_scale"],
    #             gp_output_bias=0.0,
    #             normalize_input=True,
    #             gp_cov_momentum=self.hyperparameters["gp_cov_discount_factor"],
    #             gp_cov_ridge_penalty=self.hyperparameters["gp_cov_ridge_penalty"],
    #             l2_regularization=self.hyperparameters["gp_l2_regularizer"],
    #             dataset_size = self.effective_dataset_size,
    #             scale_random_features=False,
    #             use_custom_random_features=True,
    #             custom_random_features_initializer=make_random_feature_initializer('orf'),
    #             kernel_initializer=gp_output_initializer,
    #             name=None,
    #         )
    #         logits, cov = self.output_layer(logits)
    #         # logits adjustment which is only executed in inference mode
    #         logits = LogitsAdjuster(mc_samples = self.hyperparameters["gp_mc_samples"], mean_field_factor = self.hyperparameters["gp_mean_field_factor"])(logits, cov)
    #     else: # standard linear layer instead of gp layer
    #         self.output_layer = layers.Dense(units=num_modes, 
    #             activation=None, 
    #             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01), 
    #             name="fc1000")
    #         logits = self.output_layer(logits)

    #     y_pred = layers.Activation(activation, dtype='float32')(logits)
    #     return y_pred
    
    # def create_model(self, inputs, outputs, optimizer):
    #     #model = super().create_model(inputs, outputs, optimizer)
    #     outputs = {
    #         k: layers.Activation('linear', dtype='float32', name=k)(v)
    #         for k, v in outputs.items()
    #     }

    #     model = Model(inputs=inputs, outputs=outputs)
    #     model.output_layer = self.output_layer

    #     if (
    #         'prior_model' in self.hyperparameters
    #         and self.hyperparameters['prior_model'] is not False
    #     ):
    #         self.logger.info("Running in OLA mode")
    #         model = self.ola_init(model)
    #     return model
    
    # def ola_init(self, model):
    #     """
    #     Transfer model weights and sets regularizers

    #     :param Model model: Keras model
    #     :return: Model with online Laplace regularization
    #     :rtype: Model
    #     """
    #     # Find the HeteroscedasticSNGPLayer
    #     het_sngp_layer = None
    #     for layer in model.layers:
    #         if isinstance(layer, HeteroscedasticSNGPLayer):
    #             het_sngp_layer = layer
    #             break
        
    #     if het_sngp_layer is None:
    #         raise ValueError("HeteroscedasticSNGPLayer not found in the model")
    #     # restore all weights
    #     # model.load_weights(self.hyperparameters["prior_model"])
    #     if self.hyperparameters["extractor_reg"] == "freeze":
    #         for layer in model.layers:
    #             if layer.name == "random_feature_gaussian_process":
    #                 continue
    #             else:
    #                 layer.trainable = False # freeze weights after transfer (also enables inference mode in batch normalization)
    #     else:
    #         self.ola_reg(model, reg_type=self.hyperparameters["extractor_reg"])

    #     #get old laplace approximation values
    #     rff_keys = []
    #     for name, shape in tf.train.list_variables(self.hyperparameters["prior_model"]):
    #         if "optimizer" not in name:
    #             if "gp_output_layer" in name:
    #                 beta_old_key = name
    #             elif "gp_cov_layer" in name:
    #                 precision_old_key = name
    #             elif "random_feature" in name:
    #                 rff_keys.append(name)
        
    #     # load old variables
    #     rff_bias_old = tf.train.load_variable(self.hyperparameters["prior_model"], rff_keys[0])
    #     rff_kernel_old = tf.train.load_variable(self.hyperparameters["prior_model"], rff_keys[1])
    #     beta_old = tf.train.load_variable(self.hyperparameters["prior_model"], beta_old_key)
    #     precision_old = tf.train.load_variable(self.hyperparameters["prior_model"], precision_old_key)
    #     # update fixed rff
    #     model.output_layer._random_feature.bias = rff_bias_old
    #     model.output_layer._random_feature.kernel = rff_kernel_old
    #     if self.hyperparameters["gp_reg"]:
    #         # update mode regularizer 
    #         kernel_regularizer = ModeRegularizer(weight=1./self.effective_dataset_size, mean=beta_old, precision=precision_old)
    #         model.output_layer._gp_output_layer.kernel_regularizer = kernel_regularizer
    #         model.output_layer._gp_output_layer.add_loss(lambda: kernel_regularizer(model.output_layer._gp_output_layer.kernel))
    #         # update precision matrix
    #         model.output_layer._gp_cov_layer.initial_precision_matrix = self.hyperparameters["gp_prior_decay"] * precision_old
    #     return model
    
    # def ola_reg(self, model, reg_type="equivalent"):
        # layers_to_regularize = [layer for layer in model.layers if hasattr(layer, "layer")]
        # checkpoint_keys_w = [(name, shape) for name, shape in tf.train.list_variables(self.hyperparameters["prior_model"]) if "optimizer" not in name and "/w/" in name]
        # checkpoint_keys_r = [(name, shape) for name, shape in tf.train.list_variables(self.hyperparameters["prior_model"]) if "optimizer" not in name and "/r/" in name]
        # mapping = [(idx, int(checkpoint_key_w [0].split("/")[0].replace("layer_with_weights-", ""))) for idx, checkpoint_key_w  in enumerate(checkpoint_keys_w)]
        # mapping = [(e[0],i) for i,e in enumerate(sorted(mapping, key=lambda x: x[1]))]
        # for e in mapping:
        #     w_old = tf.train.load_variable(self.hyperparameters["prior_model"], checkpoint_keys_w [e[0]][0])
        #     r = tf.train.load_variable(self.hyperparameters["prior_model"], checkpoint_keys_r[e[0]][0])
        #     print(np.min(r), "   ", np.mean(r), "    ", np.max(r))
        #     if reg_type == "empirical":
        #         layer_precision = 1./tf.math.reduce_variance(w_old) # empirical measure for layer importance
        #         weight = layer_precision/self.extractor_effective_dataset_size
        #     elif reg_type == "equivalent":
        #         weight = 1./self.extractor_effective_dataset_size
        #     elif reg_type == "fisher":
        #         weight = r/self.extractor_effective_dataset_size
        #     elif reg_type == "transfer":
        #         weight = 0
        #     reg = lambda id, weight, mean: ModeRegularizer(weight=weight, mean=mean)(layers_to_regularize[id].w)
        #     reg_loss = partial(reg, e[1], weight, w_old)
        #     with tf.keras.backend.name_scope('weight_regularizer'):
        #         layers_to_regularize[e[1]].add_loss(reg_loss)

class HetSNGPMixin():
    
    """
    Mixin for Heteroscedastic Spectral Normalized Gaussian Process outputs
    """
    def __init__(self):
        if self.hyperparameters['gp_posterior_temp'] is None:
            self.hyperparameters['gp_posterior_temp'] = (
                self.hyperparameters['batch_size']
            )
        if self.hyperparameters['extractor_posterior_temp'] is None:
            self.hyperparameters['extractor_posterior_temp'] = (
                self.hyperparameters['batch_size']
            )
        self.effective_dataset_size = (
            (self.dataset_size) /
            self.hyperparameters['gp_posterior_temp']
        )

        self.extractor_effective_dataset_size = (
            (self.dataset_size) /
            self.hyperparameters['extractor_posterior_temp']
        )

    def get_hidden(self, logits, hidden_layer_size):
        if self.hyperparameters["use_spec_norm"]:
            logits = SpectralNormalization(
                layers.Dense(hidden_layer_size, activation="relu", use_bias=False),
                norm_multiplier=self.hyperparameters["spec_norm_bound"],
            )(logits)
        else:
            logits = layers.Dense(hidden_layer_size, activation="relu", use_bias=False)(logits)
        return logits

    def get_head(self, logits, num_modes, activation):
        if self.hyperparameters["use_gp_layer"]:
            gp_output_initializer = gp_output_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
            self.output_layer = HeteroscedasticSNGPLayer(
                num_classes=num_modes,
                num_factors=num_modes,
                temperature=self.hyperparameters.get("temperature", 1.0),
                train_mc_samples=self.hyperparameters.get("train_mc_samples", 5),
                test_mc_samples=self.hyperparameters.get("test_mc_samples", 5),
                num_inducing=self.hyperparameters["gp_num_inducing"],
                gp_kernel_scale=self.hyperparameters["gp_kernel_scale"],
                gp_output_bias=0.0,
                normalize_input=True,
                gp_cov_momentum=self.hyperparameters["gp_cov_discount_factor"],
                gp_cov_ridge_penalty=self.hyperparameters["gp_cov_ridge_penalty"],
                l2_regularization=self.hyperparameters["gp_l2_regularizer"],
                dataset_size = self.effective_dataset_size,
                #gp_cov_likelihood=self.hyperparameters.get("gp_cov_likelihood", "gaussian"),
                #return_gp_cov=True,
                #return_random_features=False,
                scale_random_features=False,
                use_custom_random_features=True,
                custom_random_features_initializer=make_random_feature_initializer('orf'),
                kernel_initializer=gp_output_initializer,
                sngp_var_weight=self.hyperparameters.get("sngp_var_weight", 1.),
                het_var_weight=self.hyperparameters.get("het_var_weight", 1.),
                name="HetSNGP_layer",
            )
            logits, log_probs, probs, pred_variance = self.output_layer(logits)
        else:
            self.output_layer = layers.Dense(units=num_modes, 
                activation=None, 
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01), 
                name="fc1000")
            logits = self.output_layer(logits)

        y_pred = layers.Activation(activation, dtype='float32')(logits)
        return y_pred
    
    def create_model(self, inputs, outputs, optimizer):
        #model = super().create_model(inputs, outputs, optimizer)
        outputs = {
            k: layers.Activation('linear', dtype='float32', name=k)(v)
            for k, v in outputs.items()
        }

        model = Model(inputs=inputs, outputs=outputs)
        model.output_layer = self.output_layer
        #print(f"Loading prior model from: {self.hyperparameters['prior_model']}")
        if (
            'prior_model' in self.hyperparameters
            and self.hyperparameters['prior_model'] is not False
        ):
            self.logger.info("Running in OLA mode")
            model = self.ola_init(model)
        return model
    
    def ola_init(self, model):
        self.logger.info("Starting ola_init function")
        """
        Transfer model weights and sets regularizers

        :param Model model: Keras model
        :return: Model with online Laplace regularization
        :rtype: Model
        """
        # Find the HeteroscedasticSNGPLayer
        het_sngp_layer = None
        for layer in model.layers:
            #print(layer.name, layer.__class__.__name__)
            if isinstance(layer, HeteroscedasticSNGPLayer):
                het_sngp_layer = layer
                break
        
        if het_sngp_layer is None:
            raise ValueError("HeteroscedasticSNGPLayer not found in the model")

        if self.hyperparameters["extractor_reg"] == "freeze":
            for layer in model.layers:
                #print(layer.name)
                if not isinstance(layer, HeteroscedasticSNGPLayer):
                    layer.trainable = False  # freeze weights after transfer (also enables inference mode in batch normalization)
        else:
            self.ola_reg(model, reg_type=self.hyperparameters["extractor_reg"])

        # get old laplace approximation values
        rff_keys = []
        beta_old_key = None
        precision_old_key = None
        
        self.logger.info("Searching for variables in the prior model...")
        #print(f"2Loading prior model from: {self.hyperparameters['prior_model']}")
        for name, shape in tf.train.list_variables(self.hyperparameters["prior_model"]):
            self.logger.debug(f"Found variable: {name} with shape {shape}")
            if "optimizer" not in name:
                if "gp_output_weights" in name or "gp_output_layer" in name:
                    beta_old_key = name
                elif "gp_covariance" in name or "gp_cov_layer" in name:
                    precision_old_key = name
                elif "random_feature" in name:
                    rff_keys.append(name)
        if beta_old_key is None:
            self.logger.warning("Could not find gp_output_weights in the prior model")
        if precision_old_key is None:
            self.logger.warning("Could not find gp_covariance in the prior model")
        if len(rff_keys) < 2:
            self.logger.warning(f"Expected 2 random_feature keys, but found {len(rff_keys)}")

        # load old variables
        try:
            rff_bias_old = tf.train.load_variable(self.hyperparameters["prior_model"], rff_keys[0])
            rff_kernel_old = tf.train.load_variable(self.hyperparameters["prior_model"], rff_keys[1])
            het_sngp_layer.sngp_layer._random_feature.bias = rff_bias_old
            #print("Random Feature bias (after loading):", het_sngp_layer.sngp_layer._random_feature.bias)
            het_sngp_layer.sngp_layer._random_feature.kernel = rff_kernel_old
            #print("Random Feature kernel (after loading):", het_sngp_layer.sngp_layer._random_feature.kernel)
        except IndexError:
            self.logger.error("Could not load random feature variables")

        if beta_old_key and precision_old_key:
            beta_old = tf.train.load_variable(self.hyperparameters["prior_model"], beta_old_key)
            precision_old = tf.train.load_variable(self.hyperparameters["prior_model"], precision_old_key)
            # print(f"Loaded beta_old with shape: {beta_old.shape}")
            # print(f"Loaded precision_old with shape: {precision_old.shape}")
            
            if self.hyperparameters["gp_reg"]:
                #print("Applying GP regularization")
                # update mode regularizer 
                kernel_regularizer = ModeRegularizer(weight=1./self.effective_dataset_size, mean=beta_old, precision=precision_old)
                het_sngp_layer.sngp_layer._gp_output_layer.kernel_regularizer = kernel_regularizer
                het_sngp_layer.sngp_layer._gp_output_layer.add_loss(lambda: kernel_regularizer(het_sngp_layer.sngp_layer._gp_output_layer.kernel))
                # update precision matrix
                het_sngp_layer.sngp_layer._gp_cov_layer.initial_precision_matrix = self.hyperparameters["gp_prior_decay"] * precision_old
        else:
            self.logger.warning("Could not apply GP regularization due to missing variables")
        
        return model
    
    def ola_reg(self, model, reg_type="equivalent"):
        layers_to_regularize = [layer for layer in model.layers if hasattr(layer, "layer")]
        checkpoint_keys_w = [(name, shape) for name, shape in tf.train.list_variables(self.hyperparameters["prior_model"]) if "optimizer" not in name and "/w/" in name]
        checkpoint_keys_r = [(name, shape) for name, shape in tf.train.list_variables(self.hyperparameters["prior_model"]) if "optimizer" not in name and "/r/" in name]
        mapping = [(idx, int(checkpoint_key_w [0].split("/")[0].replace("layer_with_weights-", ""))) for idx, checkpoint_key_w  in enumerate(checkpoint_keys_w)]
        mapping = [(e[0],i) for i,e in enumerate(sorted(mapping, key=lambda x: x[1]))]
        for e in mapping:
            w_old = tf.train.load_variable(self.hyperparameters["prior_model"], checkpoint_keys_w [e[0]][0])
            r = tf.train.load_variable(self.hyperparameters["prior_model"], checkpoint_keys_r[e[0]][0])
            print(np.min(r), "   ", np.mean(r), "    ", np.max(r))
            if reg_type == "empirical":
                layer_precision = 1./tf.math.reduce_variance(w_old) # empirical measure for layer importance
                weight = layer_precision/self.extractor_effective_dataset_size
            elif reg_type == "equivalent":
                weight = 1./self.extractor_effective_dataset_size
            elif reg_type == "fisher":
                weight = r/self.extractor_effective_dataset_size
            elif reg_type == "transfer":
                weight = 0
            reg = lambda id, weight, mean: ModeRegularizer(weight=weight, mean=mean)(layers_to_regularize[id].w)
            reg_loss = partial(reg, e[1], weight, w_old)
            with tf.keras.backend.name_scope('weight_regularizer'):
                layers_to_regularize[e[1]].add_loss(reg_loss)
