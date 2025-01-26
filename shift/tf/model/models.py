'''
Collection of models.

'''
from functools import partial
import os
from pathlib import Path
import pickle

import gin
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.python.keras.losses import mean_squared_error

from shift.tf.model.covernet_utils import (
    covernet_to_class,
    covernet_to_multilabel,
    covernet_to_trajs,
    closest_trajectory,
    # closest_trajectory_2,
    #multipath_to_trajs,
)
from shift.tf.model.resnet_models import (
    ResNet50ModelFactory,
    ResNet50SNGPMixin,
    ResNet50HETSNGPMixin,

)
from shift.tf.util import (
    CyclicalPowerLearningRate,
    PolynomialWarmup,
    ConstantWarmup,
)

from shift.tf.model.due import mapping_wrapper
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class Covernet(ResNet50ModelFactory):
    """
    Model factory for CoverNet predictor

    :param int dataset_size: Number of examples in the dataset
    :param dict **kwargs: Hyperparameters
    """
    mdl_input_names = ["image", "state"]
    mdl_output_names = ["cls","trajs","cls_for_label","label","stop_comp"]

    def __init__(self, dataset_size, **kwargs):
        super().__init__(dataset_size, **kwargs)

        root = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent

        with open(
            root / "data" / f'epsilon_{self.hyperparameters["eps_set"]}.pkl', 'rb'
        ) as file:
            trajectories = pickle.load(file)
        self.lattice = np.asarray(trajectories, dtype=np.float32)

        # Note: heads do not have to be part of loss or be evaluated in metrics
        # Note: heads either differ in activation or in the passed ground truth
        self.heads = {
                "cls": ("softmax", self.lattice.shape[0], None), # ground truth is target label
                "trajs": ("softmax", None, "cls"), # ground truth is target trajectory
                "cls_for_label": ("softmax", None, "cls"), # ground truth is valid labels
                "label": ("sigmoid", None, "cls"), # ground truth is valid labels
                "stop_comp":("softmax", None, "cls"), # ground truth is valid stop line complied label
            }

    def get_loss(self):
        lattice = tf.convert_to_tensor(self.lattice.astype(np.float32))

        def constant_lattice_loss_label(y_true, y_pred):
            """
            Categorical loss with closest lattice trajectory as groundtruth. Groundtruth is assumed to be a vector of valid trajectory indices
            """
            y_true = tf.cast(
                tf.reduce_sum(tf.reduce_sum(y_true, axis=-1), axis=-1) > 0,
                tf.float32,
            )
            classification_loss = binary_crossentropy(
                y_true, tf.clip_by_value(y_pred, 1e-6, 1.0), from_logits=False
            ) * tf.cast(tf.shape(y_pred)[-1], tf.float32)
                
            return classification_loss

        def constant_lattice_loss_cls(y_true, y_pred):
            """
            Categorical loss with closest lattice trajectory as groundtruth. Single class variant.
            """
            closest_lattice_trajectory = closest_trajectory(lattice, y_true)
            classification_loss = sparse_categorical_crossentropy(
                closest_lattice_trajectory, y_pred, from_logits=False
            )
                
            return classification_loss

        if self.hyperparameters["multitask_lambda"] > 0:
            return {"cls": constant_lattice_loss_cls, "label": constant_lattice_loss_label}
        elif self.hyperparameters["multi_label"]:
            return {"label": constant_lattice_loss_label}
        else:
            return {"cls": constant_lattice_loss_cls}

    def get_mapper(self, model):
        label_mapper = covernet_to_multilabel
        traj_mapper = partial(covernet_to_trajs, lattice=self.lattice)
        class_mapper = partial(covernet_to_class, lattice=self.lattice)

        return class_mapper, label_mapper, traj_mapper

@gin.configurable
class CovernetSNGP(ResNet50SNGPMixin, Covernet):
    """
    Spectral normalized Gaussian process CoverNet model

    :param int dataset_size: Number of examples in the dataset
    :param int batch_size: Number of example sper batch
    :param float lr: learning rate
    :param float momentum: momentum
    :param int eps_set: Epsilon of the Covernet trajectory set (2,4,8)
    :param float spectral_norm: Spectral normalization constant
    :param int num_inducing: Number of kernel inducing points
    :param float gp_kernel_scale: Scale factor of the RBF kernel
    :param bool multi_label: If true, assumes multiple, valid trajectories per example
    """

    def __init__(
        self,
        dataset_size,
        batch_size,
        lr=1e-4,
        momentum=0.9,
        eps_set=4,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        use_spec_norm=True,
        spec_norm_bound=6.,
        spec_norm_iteration=1,
        use_gp_layer=True,
        gp_mean_field_factor = np.pi/8, # temperature scaling
        gp_mc_samples = 0,
        gp_num_inducing=1024,
        gp_kernel_scale=1.0,
        gp_l2_regularizer=0.0, # regularizer output weights
        gp_cov_discount_factor=-1., # moving average momentum, -1 for exact computation
        gp_cov_ridge_penalty=1., # ensures stability of matrix inverse if number of samples small
        multi_label=False,
        prior_model=False,
        extractor_reg="freeze",
        gp_reg = True,
        extractor_posterior_temp=None,
        gp_posterior_temp=None, # weighting of prior quadratic penalty, None = #batches, 1. = #samples
        gp_prior_decay=1., # 0. < decay < 1., =1. for no decay
        multitask_lambda=0.,
    ):
        Covernet.__init__(
            self,
            dataset_size=dataset_size,
            batch_size=batch_size,
            lr=lr,
            momentum=momentum,
            eps_set=eps_set,
            use_spec_norm=use_spec_norm,
            spec_norm_bound=spec_norm_bound,
            spec_norm_iteration=spec_norm_iteration,
            use_gp_layer=use_gp_layer,
            gp_mean_field_factor = gp_mean_field_factor,
            gp_mc_samples = gp_mc_samples,
            gp_num_inducing=gp_num_inducing,
            gp_kernel_scale=gp_kernel_scale,
            gp_l2_regularizer=gp_l2_regularizer,
            gp_cov_discount_factor=gp_cov_discount_factor,
            gp_cov_ridge_penalty=gp_cov_ridge_penalty,
            multi_label=multi_label,
            prior_model=prior_model,
            extractor_reg=extractor_reg,
            gp_reg = gp_reg,
            extractor_posterior_temp=extractor_posterior_temp,
            gp_posterior_temp=gp_posterior_temp,
            gp_prior_decay=gp_prior_decay,
            multitask_lambda=multitask_lambda,
        )
        ResNet50SNGPMixin.__init__(self)

@gin.configurable
class CovernetHETSNGP(ResNet50HETSNGPMixin, Covernet):
    """
    heteroscedastic Spectral normalized Gaussian process CoverNet model

    :param int dataset_size: Number of examples in the dataset
    :param int batch_size: Number of example sper batch
    :param float lr: learning rate
    :param float momentum: momentum
    :param int eps_set: Epsilon of the Covernet trajectory set (2,4,8)
    :param float spectral_norm: Spectral normalization constant
    :param int num_inducing: Number of kernel inducing points
    :param float gp_kernel_scale: Scale factor of the RBF kernel
    :param bool multi_label: If true, assumes multiple, valid trajectories per example
    """

    def __init__(
        self,
        dataset_size,
        batch_size,
        lr=1e-4,
        momentum=0.9,
        eps_set=4,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        use_spec_norm=True,
        spec_norm_bound=6.,
        spec_norm_iteration=1,
        use_gp_layer=True,
        gp_mean_field_factor = np.pi/8, # temperature scaling
        gp_mc_samples = 0,
        gp_num_inducing=1024,
        gp_kernel_scale=1.0,
        gp_l2_regularizer=0.0, # regularizer output weights
        gp_cov_discount_factor=-1., # moving average momentum, -1 for exact computation
        gp_cov_ridge_penalty=1., # ensures stability of matrix inverse if number of samples small
        multi_label=False,
        prior_model=False,
        extractor_reg="freeze",
        gp_reg = True,
        extractor_posterior_temp=None,
        gp_posterior_temp=None, # weighting of prior quadratic penalty, None = #batches, 1. = #samples
        gp_prior_decay=1., # 0. < decay < 1., =1. for no decay
        multitask_lambda=0.,
        sngp_var_weight=1.,
        het_var_weight=1.,
        temperature=1.0, # Temperature parameter
        train_mc_samples=10, # Number of MC samples for training
        test_mc_samples=10, # Number of MC samples for testing
    ):
        Covernet.__init__(
            self,
            dataset_size=dataset_size,
            batch_size=batch_size,
            lr=lr,
            momentum=momentum,
            eps_set=eps_set,
            use_spec_norm=use_spec_norm,
            spec_norm_bound=spec_norm_bound,
            spec_norm_iteration=spec_norm_iteration,
            use_gp_layer=use_gp_layer,
            gp_mean_field_factor = gp_mean_field_factor,
            gp_mc_samples = gp_mc_samples,
            gp_num_inducing=gp_num_inducing,
            gp_kernel_scale=gp_kernel_scale,
            gp_l2_regularizer=gp_l2_regularizer,
            gp_cov_discount_factor=gp_cov_discount_factor,
            gp_cov_ridge_penalty=gp_cov_ridge_penalty,
            multi_label=multi_label,
            prior_model=prior_model,
            extractor_reg=extractor_reg,
            gp_reg = gp_reg,
            extractor_posterior_temp=extractor_posterior_temp,
            gp_posterior_temp=gp_posterior_temp,
            gp_prior_decay=gp_prior_decay,
            multitask_lambda=multitask_lambda,
            sngp_var_weight = sngp_var_weight,
            het_var_weight = het_var_weight,
            temperature = temperature,
            train_mc_samples = train_mc_samples,
            test_mc_samples = test_mc_samples,

        )
        ResNet50HETSNGPMixin.__init__(self)
