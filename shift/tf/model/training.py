'''
Model training classes.

'''
from datetime import datetime
import logging
import numbers
import os
from pathlib import Path
import re
import shutil
from functools import partial
import pickle

import gin
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import ray
from ray import tune
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow_addons.metrics.utils import MeanMetricWrapper
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction.helper import convert_local_coords_to_global
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe


import os
from shift.tf.dataloader.NuScenesDataLoader import NuscenesDataset
from shift.tf.util import (
    TFSaver,
    MaxMetricWrapper,
    MultiOutputMeanMetricWrapper,
    CollectingMetricWrapper,
    LastMetricWrapper,
    TensorBoardMod,
    DelayedEarlyStopping,
    ExpTracker
)

from shift.tf.model.sngp import SNGPResetCovariance
from shift.tf.fixes import TensorflowFix
from shift.tf.metrics import (
    Joining,
    drivable_area_compliance,
    displacement_error,
    stop_line_compliance,
    #brier_score,
    hit_rate,
    class_rank,
    class_nll,
    #relative_displacement_error,
    #predictive_entropy,
    MappedSparseCategoricalAccuracy,
    #MappedECE,
    #MappedPrecision,
    #MappedRecall,
    #MappedBinaryAccuracy,
)
import numpy as np
import tensorflow as tf
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction.helper import convert_local_coords_to_global
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
import matplotlib.pyplot as plt

@gin.configurable
class Trainer:
    """
    Basic trainer class. Handles model training, loading and runtime logging.

    :param model_factory: A model factory class, derived from :class:Covernet
    :param Path dir: Root folder for the experiment
    :param str,None: Current time string, if None, defaults to now
    :param str name: Model name, usually model_i, with i as the index
    :param int start_epoch: Number of the first epoch
    :param int epochs: Number of total epochs
    :param int batch_size: Number of examples per batch
    :param Path load_model: Model file path to load (as initialization)
    :param Path resume_model: Model file path to load (as resume)
    :param bool is_tune: If True, several logging methods are disabled (handled via ray)
    :param bool multi_label: If True, a multi label loss is used (BCE instead of CE)
    :param float multitask_lambda: If > 0, multitask output is assumed
    :param bool mixed_precision: If True, mixed float16 is used.
    :param dict kwargs: Additional hyperparameters, passed to the model factory.
    """

    def __init__(
        self,
        model_factory,
        dir,
        current_time=None,
        name="model_0",
        start_epoch=0,
        epochs=20,
        batch_size=64,
        load_model=None,
        resume_model=None,
        is_tune=False,
        multi_label=False,
        multitask_lambda=0.0,
        reg_lambda=0.0,
        cls_lambda=1.0,
        mixed_precision=True,
        early_stop_delay=300,
        early_stop_patience=100,
        save_delay=0,
        post_evaluate=True,
        post_calibration=False,
        cal_lr=0.001,
        cal_l2=0.001,
        cal_epochs=500,
        cal_batch_size=128,
        fisher_samples=0,
        fisher_type="pred",
        save_test_preds=False,
        track_file = None,
        description = "",
        jit = True,
        **kwargs,
    ):
        if jit and not 'prior_model' in kwargs:
            tf.config.optimizer.set_jit(True)

        if len(tf.config.list_physical_devices('GPU')) >= 1 and mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

        mdl_input_names = model_factory.mdl_input_names
        mdl_output_names = model_factory.mdl_output_names
        with gin.config_scope('train'):
            self.train_dataset = NuscenesDataset().get_dataset(mdl_input_names, mdl_output_names, shuffle=True)

        with gin.config_scope('val'):
            self.val_dataset = NuscenesDataset().get_dataset(mdl_input_names, mdl_output_names)

        with gin.config_scope('test'):
            self.test_dataset = NuscenesDataset().get_dataset(mdl_input_names, mdl_output_names)

        if "CovernetDUE" in str(model_factory):
            self.factory = model_factory(
                # dataset = init_dataset,
                dataset_size = len(self.train_dataset),
                batch_size=batch_size,
                multi_label=multi_label,
                multitask_lambda=multitask_lambda,
                **kwargs,
            )
        else:
            self.factory = model_factory(
                dataset_size = len(self.train_dataset),
                batch_size=batch_size,
                multi_label=multi_label,
                multitask_lambda=multitask_lambda,
                **kwargs,
            )

        self.name = name
        self.current_time = current_time

        self.exp_dir = dir
        self.logger = logging.getLogger('trainer')

        self.hyperparameters = self.factory.hyperparameters
        self.hyperparameters.update({"batch_size": batch_size})
        self.hyperparameters.update({"mixed_precision": mixed_precision})
        self.hyperparameters.update({"early_stop_delay": early_stop_delay})
        self.hyperparameters.update({"early_stop_patience": early_stop_patience})
        self.is_tune = is_tune

        self.start_epoch = start_epoch
        self.epochs = epochs
        self.batch_size = batch_size

        self.multi_label = multi_label
        self.multitask_lambda = multitask_lambda
        self.reg_lambda = reg_lambda
        self.cls_lambda = cls_lambda
        self.post_evaluate = post_evaluate

        self.load_model = load_model
        self.resume_model = resume_model

        self.early_stop_delay = early_stop_delay
        self.early_stop_patience = early_stop_patience

        self.fisher_samples = fisher_samples
        self.fisher_type = fisher_type

        self.save_delay = save_delay
        self.post_calibration = post_calibration
        self.cal_lr = cal_lr
        self.cal_l2 = cal_l2
        self.cal_epochs = cal_epochs
        self.cal_batch_size = cal_batch_size

        self.save_test_preds = save_test_preds
        self.track_file = Path(track_file) if track_file is not None else track_file
        self.description= description

    def _build_metrics(self):
        cls_metrics = self._build_multi_class_metrics()
        cls_for_label_metrics = self._build_class_for_label_metrics()
        label_metrics = self._build_multi_label_metrics()
        trajs_metrics = self._build_trajectory_metrics()
        cls_for_label_metrics_stop=self._build_class_for_label_metrics_stop()

        if isinstance(self.optimizer.lr, LearningRateSchedule):
            def lr_metric(y_true, y_pred):
                return self.optimizer.lr(self.optimizer.iterations)
        else:
            def lr_metric(y_true, y_pred):
                return self.optimizer.lr

        joined_metrics = [MultiOutputMeanMetricWrapper(lr_metric, name="lr")]
        if hasattr(self.model, 'loss_scaler'):
            joined_metrics = joined_metrics + self._build_vi_metrics()

        # in case of MultiPath model the trajectory metrics are jointly evaluated on the cls and reg head
        if "reg" in self.model.output_names:
            joined_metrics = joined_metrics + trajs_metrics
            joinings = [Joining(joined_metrics, ("cls","reg"))]
            return {"cls": cls_metrics, "cls_for_label": cls_for_label_metrics,"stop_comp": cls_for_label_metrics_stop, "label": label_metrics, "joined": joinings}
        else:
            joinings = [Joining(joined_metrics, ("cls",))]
            return {"cls": cls_metrics, "cls_for_label": cls_for_label_metrics,"stop_comp": cls_for_label_metrics_stop, "label": label_metrics, "trajs": trajs_metrics, "joined": joinings}

        

    def _build_class_for_label_metrics(self):
        "Adds class for label metrics to the keras fit call"
        metrics = [
            MeanMetricWrapper(
                drivable_area_compliance,
                name="dac",
                mapper=self.mapper_label,
            ),
        ]

        return metrics
    #for stop line compliance metric calculation
    def _build_class_for_label_metrics_stop(self):
        "Adds class for label metrics to the keras fit call"
        metrics = [
            MeanMetricWrapper(
                stop_line_compliance,
                name="slc",
                mapper=self.mapper_label,
            ),
        ]

        return metrics

    def _build_multi_label_metrics(self):
        """
        Adds multi label metrics to the keras fit call
        """

        def mapped_BCE(y_true, y_pred):
            return binary_crossentropy(
                *self.mapper_label(y_true, y_pred), from_logits=False
            )

        metrics = [
            MeanMetricWrapper(
                mapped_BCE,
                name='nll',
            ),
            MeanMetricWrapper(
                class_nll,
                mapper=self.mapper_label,
                cls_idx=0,
                name='neg_nll',
            ),
            MeanMetricWrapper(
                class_nll,
                mapper=self.mapper_label,
                cls_idx=1,
                name='pos_nll',
            ),
            #MappedBinaryAccuracy(mapper=self.mapper_label, name='bacc'),
            #MappedPrecision(mapper=self.mapper_label, name='prec'),
            #MappedRecall(mapper=self.mapper_label, name='rec'),
            #MappedECE(mapper=self.mapper_label, name='ece'),
        ]

        return metrics

    def _build_trajectory_metrics(self):
        
        metrics = [
            CollectingMetricWrapper(
                displacement_error,
                len(self.train_dataset),
                final_only=False,
                k=1,
                mapper=self.mapper_trajs,
                name='ade1',
            ),
            CollectingMetricWrapper(
                displacement_error,
                len(self.train_dataset),
                final_only=False,
                k=5,
                mapper=self.mapper_trajs,
                name='ade5',
            ),
            CollectingMetricWrapper(
                displacement_error,
                len(self.train_dataset),
                final_only=False,
                k=10,
                mapper=self.mapper_trajs,
                name='ade10',
            ),
            CollectingMetricWrapper(
                displacement_error,
                len(self.train_dataset),
                final_only=False,
                k=15,
                mapper=self.mapper_trajs,
                name='ade15',
            ),
            CollectingMetricWrapper(
                displacement_error,
                len(self.train_dataset),
                final_only=False,
                k=20,
                mapper=self.mapper_trajs,
                name='ade20',
            ),
            CollectingMetricWrapper(
                displacement_error,
                len(self.train_dataset),
                final_only=True,
                mapper=self.mapper_trajs,
                name='fde',
            ),
            CollectingMetricWrapper(
                displacement_error,
                len(self.train_dataset),
                k=20,
                final_only=True,
                mapper=self.mapper_trajs,
                name='fde20',
            ),
            # CollectingMetricWrapper(
            #     relative_displacement_error,
            #     len(self.train_dataset),
            #     final_only=False,
            #     k=5,
            #     mapper=self.mapper_trajs,
            #     name="rel_ade5"
            # ),
            MultiOutputMeanMetricWrapper(
                hit_rate,
                final_only=False,
                k=5,
                d=2,
                mapper=self.mapper_trajs,
                name="hitrate52"
            ),
        ]
        return metrics
                
    def _build_multi_class_metrics(self):
        """
        Adds class label metrics to the keras fit call
        """

        def mapped_sparse_CE(y_true, y_pred):
            return sparse_categorical_crossentropy(
                *self.mapper_class(y_true, y_pred), from_logits=False
            )

        metrics = [
            CollectingMetricWrapper(
                class_rank,
                len(self.train_dataset),
                mapper=self.mapper_class,
                name='rnk',
            ),
            MeanMetricWrapper(
                mapped_sparse_CE,
                name='nll',
            ),
            # MeanMetricWrapper(
            #     brier_score,
            #     mapper=self.mapper_class,
            #     name="brier-score"
            # ),
            #MappedECE(mapper=self.mapper_class, name='ece'),
            MappedSparseCategoricalAccuracy(mapper=self.mapper_class, name="acc"),
            # CollectingMetricWrapper(
            #     predictive_entropy,
            #     len(self.train_dataset),
            #     name="pred_entropy",
            # ),
        ]

        return metrics
    
    def _per_layer_load_model_mismatched(self, load_model):
        """
        Loads VI layers as deterministic means and deterministic layers as VI means. Only loads mismatched layers, has to be called after Checkpoint.restore
        
        :param str,Path load_model: the model to load
        """
        collected_variables = []
        collected_variables_is_vi =[]
        for layer in self.model.layers:
            variables = (
                layer.trainable_variables + layer.non_trainable_variables
            )
            if len(variables) > 0:
                collected_variables.append(
                    {
                        variable.name.split("/")[1].replace(":0", ""): variable
                        for variable in variables
                        if not 'stddev' in variable.name
                    }
                )
                collected_variables_is_vi.append('Flipout' in str(layer.__class__))
        checkpoint_keys = []
        for name, shape in tf.train.list_variables(load_model):
            if not 'optimizer' in name and len(shape) > 0:
                checkpoint_keys.append(name)

        for name in checkpoint_keys: 
            elements = name.split("/")
            layer_index = int(elements[0].replace("layer_with_weights-", ""))
            variable = collected_variables[layer_index]
                              
            if collected_variables_is_vi[layer_index] and "kernel_initializer" not in name:          
                self.logger.info(f'Loading {name} as VI mean init')
            elif not collected_variables_is_vi[layer_index]:
                if "stddev" in name:
                    continue      
                if elements[1] == "kernel_initializer":
                    elements[1] = "kernel"
                self.logger.info(f'Loading {name} VI means as deterministic') 
            else:
                self.logger.info(f'Skipping {name}, should already be loaded') 
                continue
                
            variable = variable[elements[1]]                                              
            data = tf.train.load_variable(load_model, name)
            variable.assign(data)
                     
    def _load_model(self, load_model, isResume):
        """
        Loads a model file for resume or as initializer.

        :param Str,Path load_model: Path to the model to load. A checkpoint, Model- or Model_ep file (without suffix)
        :param bool isResume: If True, optimizer and start_epoch is not loaded
        """

        if 'checkpoint' in Path(load_model).name:
            load_model = str(
                list(Path(load_model).parent.glob("*.index"))[0].with_suffix('')
            )
            if isResume:
                self.start_epoch = (
                    int(re.search(f'Model-(\d+).*', Path(load_model).name).group(1)) + 1
                )
        elif 'Model-' in Path(load_model).name:
            if isResume:
                self.start_epoch = (
                    int(re.search(f'Model-(\d+).*', Path(load_model).name).group(1)) + 1
                )
        elif 'Model_ep' in Path(load_model).name:
            if isResume:
                self.start_epoch = (
                    int(re.search(f'Model_ep(\d+)-.*', Path(load_model).name).group(1))
                    + 1
                )
        else:
            raise AttributeError(f'{Path(load_model).name} is not a valid format')

        if not isResume:
            status = tf.train.Checkpoint(self.model).restore(load_model)
        else:
            status = tf.train.Checkpoint(self.model, optimizer=self.optimizer).restore(
                load_model
            )

        try:
            status.assert_existing_objects_matched()
            self.logger.info(f'Loaded {load_model} as full match')
        except Exception as e:
            self._per_layer_load_model_mismatched(load_model)

    def build(self):
        """
        Build the model (compile), adds metrics and loads prior data.
        """

        (
            model,
            loss,
            optimizer,
            mapper_trajs,
            mapper_class,
            mapper_label,
        ) = self.factory.build()
        #model.summary()

        self.model = model
        self.optimizer = optimizer
        self.loss = loss

        self.mapper_label = mapper_label
        self.mapper_class = mapper_class
        self.mapper_trajs = mapper_trajs

        if not self.is_tune:
            if self.current_time is None:
                self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = self.exp_dir / 'logs' / self.current_time / self.name
            self.mdl_dir = self.exp_dir / 'models' / self.current_time / self.name
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.mdl_dir.mkdir(parents=True, exist_ok=True)
            tf.print(f"Model checkpoints will be saved in {self.mdl_dir}")
            self.trial_name = f'logs/{self.current_time}/{self.name}/'
        else:
            self.checkpointer = tf.train.Checkpoint(self.model)

        if self.resume_model is not None:
            load_model = self.resume_model
            isResume = True
        elif self.load_model is not None:
            load_model = self.load_model
            isResume = False
        else:
            load_model = None

        if load_model:
            self._load_model(load_model, isResume)

        # loss weight computation for MTP and Multitask
        sum_lambda = self.cls_lambda + self.reg_lambda + self.multitask_lambda
        loss_weights = {"cls": self.cls_lambda / sum_lambda }
        if self.reg_lambda > 0.0:
            loss_weights["reg"] = self.reg_lambda / sum_lambda
        if self.multitask_lambda > 0.0:
            loss_weights["label"] =  self.multitask_lambda / sum_lambda

        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self._build_metrics(), loss_weights = loss_weights,
        )

    def get_callbacks(self, is_test=False, test_epoch=0):
        """
        Creates Keras Callbacks. If self.is_tune is True, this will generate an empty list.

        :param bool is_test: If True, a TensorBoard logger writing test metrics, is created.
        :param int test_epoch: Current epoch, for eval only.
        :return: Callbacks
        :rtype: list[callback]
        """
        if self.is_tune:
            callbacks = [TensorflowFix()]
        else:
            self.tensorboard_cb = TensorBoardMod(
                log_dir=self.log_dir,
                epoch=test_epoch,
                update_freq='epoch',
                profile_batch=0,
                write_graph=False,
                is_test=is_test,
            )
            self.tensorboard_cb.set_model(self.model)
            if not is_test:
                with self.tensorboard_cb._train_writer.as_default():
                    tf.summary.text("config", gin.config.config_str(), step=0)
            with self.tensorboard_cb._val_writer.as_default():
                tf.summary.text("config", gin.config.config_str(), step=0)

            if self.multi_label: # label head provides loss
                measures = ['epoch', 'val_label_loss', 'loss']
                val_nll = ["val_label_nll"]
            else: # otherwise cls head provides loss (in MultiPath case this is a simplification)
                measures = ['epoch', 'val_cls_loss', 'loss']
                val_nll = ["val_cls_nll"]

            is_min = [False, True, True]
            if hasattr(self.model, 'loss_scaler'):
                measures = measures + val_nll
                is_min = is_min + [True, True]
            callbacks = [
                TensorflowFix(),
                self.tensorboard_cb,
                TFSaver(
                    self.mdl_dir,
                    self.model,
                    self.model.optimizer,
                    measures=measures,
                    is_min=is_min,
                    last_n=2,
                    start_delay=self.save_delay
                ),
                hp.KerasCallback(
                    self.tensorboard_cb._val_writer,
                    self.hyperparameters,
                    self.trial_name + ("test" if is_test else "validation"),
                ),
            ]
            if not is_test:
                callbacks.append(
                    hp.KerasCallback(
                        self.tensorboard_cb._train_writer,
                        self.hyperparameters,
                        self.trial_name + "train",
                    )
                )
                if hasattr(self.model, "output_layer"):
                    if hasattr(self.model.output_layer, "gp_cov_momentum"):
                        if self.model.output_layer.gp_cov_momentum == -1:
                            callbacks.append(SNGPResetCovariance())
            else:
                callbacks.insert(1,ExpTracker(
                    file=self.track_file if self.track_file is not None else self.log_dir/"results.csv",
                    description=self.description,
                    # extractor_posterior_temp=self.config.get('extractor_posterior_temp'),
                    # gp_posterior_temp= self.gp_posterior_temp,
                    # spec_norm_bound=self.spec_norm_bound,
                    # gp_kernel_scale=self.gp_kernel_scale,
                    ID=self.log_dir.parent.name,
                    factory=self.factory.__class__.__name__,
                    exp_type=self.log_dir.parent.parent.parent.parent.parent.name,
                    epsilon=self.log_dir.parent.parent.parent.parent.name,
                    epoch=test_epoch,
                    test_data=str(gin.config._CONFIG[("test", "shift.tf.dataloader.NuScenesDataLoader.NuscenesDataset")].get("limit", ""))
                    )
                )
            callbacks.append(
                DelayedEarlyStopping(
                    start_delay=self.early_stop_delay,
                    patience=self.early_stop_patience,
                    monitor=val_nll[0],
                )
            )

        return callbacks

    def eval(self, epoch):
        """
        Runs model evaluation.
        """
        test_dataset = self.test_dataset.batch(self.batch_size, drop_remainder=True).prefetch(
            tf.data.AUTOTUNE
        )
        callbacks = self.get_callbacks(is_test=True, test_epoch=epoch)
        if self.post_calibration:
            self.model = self.post_calibrate(lr=self.cal_lr, l2_lambda=self.cal_l2, batch_size=self.cal_batch_size, epochs=self.cal_epochs)

        for cb in callbacks:
            if isinstance(cb, hp.KerasCallback):
                cb.on_train_begin()
        
        if self.save_test_preds:
            predictions = self.model.predict(test_dataset)
            pred_path = self.log_dir/ "test_preds.pkl"
            with open(pred_path, "wb") as f:
                pickle.dump(predictions,f)

        self.model.evaluate(test_dataset, callbacks=callbacks)

        if self.post_evaluate:
            self.post_eval(test_dataset, callbacks)

        for cb in callbacks:
            if isinstance(cb, hp.KerasCallback):
                cb.on_train_end()

    def post_eval(self, dataset, callbacks):
        print("We are entering in plotting the prediction mode !!")
        """
        Runs post-evaluation for a non-MTP model to compute the DAC metric and log it in TensorBoard.
        """
        predictions_for_plotting = []
        
        # Initialize dataloader with correct split and config
        with gin.config_scope('test'):
            dataloader = NuscenesDataset()
            
        # Get the raw dataset that includes tokens
        raw_dataset = dataloader.get_dataset(['image', 'state', 'instance_token', 'sample_token'], ['cls'])
        raw_dataset = raw_dataset.batch(self.batch_size, drop_remainder=True)
        
        # Iterate through both datasets simultaneously
        for (model_inputs, _), (raw_inputs, _) in zip(dataset, raw_dataset):
            # Get model predictions
            predictions = self.model(model_inputs)
            
            if 'cls' not in predictions:
                print(f"Warning: 'cls' not in predictions. Keys: {predictions.keys()}")
                continue
                
            # Get most likely trajectories
            most_likely_trajectories = tf.argmax(predictions['cls'], axis=-1)
            batch_trajectories = tf.gather(self.factory.lattice, most_likely_trajectories)
            
            # Get tokens from raw dataset
            instance_tokens = raw_inputs['instance_token']
            sample_tokens = raw_inputs['sample_token']
            
            # Store predictions for plotting
            for i in range(len(instance_tokens)):
                predictions_for_plotting.append({
                    'instance_token': instance_tokens[i].numpy().decode(),
                    'sample_token': sample_tokens[i].numpy().decode(),
                    'predicted_trajectory': batch_trajectories[i].numpy()
                })
        
        # Plot predictions
        self.plot_predictions_on_map(predictions_for_plotting, dataloader)

        # for plot with history in png format for easier viewing
    def plot_predictions_on_map(self, predictions_for_plotting, dataloader):
        """
        Plots predictions with past history, ground truth and predicted trajectories
        """
        plot_dir = "/app/plot_w_h_png"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Remove SVG-specific parameter as we're using PNG
        
        for idx, prediction in enumerate(predictions_for_plotting[:9041], 1):
            instance_token = prediction['instance_token']
            sample_token = prediction['sample_token']
            predicted_trajectory = prediction['predicted_trajectory']
            
            # Create figure with higher DPI for better PNG quality
            fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            
            # Get map and annotation data
            map_name = dataloader.helper.get_map_name_from_sample_token(sample_token)
            nusc_map = NuScenesMap(map_name=map_name, dataroot=dataloader.data_dir)
            starting_annotation = dataloader.helper.get_sample_annotation(instance_token, sample_token)
            
            # Render map patch
            patch_box = (
                starting_annotation['translation'][0] - 25,
                starting_annotation['translation'][1] - 25,
                starting_annotation['translation'][0] + 25,
                starting_annotation['translation'][1] + 25,
            )
            
            fig, ax = nusc_map.render_map_patch(
                box_coords=patch_box,
                layer_names=['drivable_area', 'road_segment', 'lane', 'ped_crossing',
                            'walkway', 'stop_line', 'road_divider', 'lane_divider'],
                alpha=0.2,
                figsize=(3, 3),
                render_egoposes_range=False,
            )
            
            # Remove legend
            if ax.get_legend():
                ax.get_legend().remove()

            # Get and plot past trajectory (history) for main agent
            past_trajectory = dataloader.helper.get_past_for_agent(
                instance_token,
                sample_token,
                seconds=2,  # 2 seconds of history
                in_agent_frame=False,
                just_xy=True
            )
            if len(past_trajectory) > 0:
                ax.plot(
                    past_trajectory[:, 0],
                    past_trajectory[:, 1],
                    color='green',  # Bright green for history
                    linestyle='--',
                    linewidth=2.0,
                    alpha=0.8,
                    zorder=610,
                    label='History'
                )

            # Plot predicted trajectory
            predicted_trajectory_global = convert_local_coords_to_global(
                predicted_trajectory,
                starting_annotation['translation'],
                Quaternion(starting_annotation['rotation']),
            )
            ax.plot(
                predicted_trajectory_global[:, 0],
                predicted_trajectory_global[:, 1],
                color='red',
                linestyle='-.',
                linewidth=2.0,
                alpha=0.8,
                zorder=620,
                label='Predicted'
            )

            # Plot ground truth
            gt_trajectory = dataloader.helper.get_future_for_agent(
                instance_token,
                sample_token,
                seconds=6,
                in_agent_frame=False,
                just_xy=True
            )
            ax.plot(
                gt_trajectory[:, 0],
                gt_trajectory[:, 1],
                color='blue',
                linewidth=2.0,
                path_effects=[pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()],
                label='Ground Truth'
            )

            # Plot main vehicle
            vehicle_box = Box(
                center=starting_annotation['translation'],
                size=starting_annotation['size'],
                orientation=Quaternion(starting_annotation['rotation']),
            )
            corners = vehicle_box.corners()
            ax.fill(
                corners[0, [0, 1, 5, 4, 0]],
                corners[1, [0, 1, 5, 4, 0]],
                color='black',
                alpha=0.7,
                zorder=630
            )

            # Plot other agents with their history
            all_agents = dataloader.helper.get_annotations_for_sample(sample_token)
            for agent in all_agents:
                if agent['instance_token'] != instance_token:
                    # Plot agent box
                    agent_box = Box(
                        center=agent['translation'],
                        size=agent['size'],
                        orientation=Quaternion(agent['rotation']),
                    )
                    corners = agent_box.corners()
                    ax.plot(
                        corners[0, [0, 1, 5, 4, 0]],
                        corners[1, [0, 1, 5, 4, 0]],
                        color='magenta',
                        linewidth=1,
                        alpha=0.5
                    )
                    
                    # Plot other agents' history
                    other_past = dataloader.helper.get_past_for_agent(
                        agent['instance_token'],
                        sample_token,
                        seconds=2,
                        in_agent_frame=False,
                        just_xy=True
                    )
                    if len(other_past) > 0:
                        ax.plot(
                            other_past[:, 0],
                            other_past[:, 1],
                            color='yellow',  # Yellow for other agents' history
                            linestyle='--',
                            linewidth=1,
                            alpha=0.5
                        )

            ax.set_aspect('equal')
            
            # Save figure as PNG instead of SVG
            plt.savefig(
                os.path.join(plot_dir, f"{idx}.png"),  # Changed extension to .png
                format='png',
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300
            )
            plt.close(fig)

        print(f"Saved {min(9042, len(predictions_for_plotting))} sample images.")
    # for svg plot for high wuality image with history
    # def plot_predictions_on_map(self, predictions_for_plotting, dataloader):
    #     """
    #     Plots predictions with past history, ground truth and predicted trajectories
    #     """
    #     plot_dir = "/app/plot_w_h"
    #     os.makedirs(plot_dir, exist_ok=True)
        
    #     # Set plot style
    #     plt.rcParams['svg.fonttype'] = 'none'
        
    #     for idx, prediction in enumerate(predictions_for_plotting[:9041], 1):
    #         instance_token = prediction['instance_token']
    #         sample_token = prediction['sample_token']
    #         predicted_trajectory = prediction['predicted_trajectory']
            
    #         # Create figure
    #         fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
            
    #         # Get map and annotation data
    #         map_name = dataloader.helper.get_map_name_from_sample_token(sample_token)
    #         nusc_map = NuScenesMap(map_name=map_name, dataroot=dataloader.data_dir)
    #         starting_annotation = dataloader.helper.get_sample_annotation(instance_token, sample_token)
            
    #         # Render map patch
    #         patch_box = (
    #             starting_annotation['translation'][0] - 25,
    #             starting_annotation['translation'][1] - 25,
    #             starting_annotation['translation'][0] + 25,
    #             starting_annotation['translation'][1] + 25,
    #         )
            
    #         fig, ax = nusc_map.render_map_patch(
    #             box_coords=patch_box,
    #             layer_names=['drivable_area', 'road_segment', 'lane', 'ped_crossing',
    #                         'walkway', 'stop_line', 'road_divider', 'lane_divider'],
    #             alpha=0.2,
    #             figsize=(3, 3),
    #             render_egoposes_range=False,
    #         )
            
    #         # Remove legend
    #         if ax.get_legend():
    #             ax.get_legend().remove()

    #         # Get and plot past trajectory (history) for main agent
    #         past_trajectory = dataloader.helper.get_past_for_agent(
    #             instance_token,
    #             sample_token,
    #             seconds=2,  # 2 seconds of history
    #             in_agent_frame=False,
    #             just_xy=True
    #         )
    #         if len(past_trajectory) > 0:
    #             ax.plot(
    #                 past_trajectory[:, 0],
    #                 past_trajectory[:, 1],
    #                 color='green',  # Bright green for history
    #                 linestyle='--',
    #                 linewidth=2.0,
    #                 alpha=0.8,
    #                 zorder=610,
    #                 label='History'
    #             )

    #         # Plot predicted trajectory
    #         predicted_trajectory_global = convert_local_coords_to_global(
    #             predicted_trajectory,
    #             starting_annotation['translation'],
    #             Quaternion(starting_annotation['rotation']),
    #         )
    #         ax.plot(
    #             predicted_trajectory_global[:, 0],
    #             predicted_trajectory_global[:, 1],
    #             color='red',
    #             linestyle='-.',
    #             linewidth=2.0,
    #             alpha=0.8,
    #             zorder=620,
    #             label='Predicted'
    #         )

    #         # Plot ground truth
    #         gt_trajectory = dataloader.helper.get_future_for_agent(
    #             instance_token,
    #             sample_token,
    #             seconds=6,
    #             in_agent_frame=False,
    #             just_xy=True
    #         )
    #         ax.plot(
    #             gt_trajectory[:, 0],
    #             gt_trajectory[:, 1],
    #             color='blue',
    #             linewidth=2.0,
    #             path_effects=[pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()],
    #             label='Ground Truth'
    #         )

    #         # Plot main vehicle
    #         vehicle_box = Box(
    #             center=starting_annotation['translation'],
    #             size=starting_annotation['size'],
    #             orientation=Quaternion(starting_annotation['rotation']),
    #         )
    #         corners = vehicle_box.corners()
    #         ax.fill(
    #             corners[0, [0, 1, 5, 4, 0]],
    #             corners[1, [0, 1, 5, 4, 0]],
    #             color='black',
    #             alpha=0.7,
    #             zorder=630
    #         )

    #         # Plot other agents with their history
    #         all_agents = dataloader.helper.get_annotations_for_sample(sample_token)
    #         for agent in all_agents:
    #             if agent['instance_token'] != instance_token:
    #                 # Plot agent box
    #                 agent_box = Box(
    #                     center=agent['translation'],
    #                     size=agent['size'],
    #                     orientation=Quaternion(agent['rotation']),
    #                 )
    #                 corners = agent_box.corners()
    #                 ax.plot(
    #                     corners[0, [0, 1, 5, 4, 0]],
    #                     corners[1, [0, 1, 5, 4, 0]],
    #                     color='magenta',
    #                     linewidth=1,
    #                     alpha=0.5
    #                 )
                    
    #                 # Plot other agents' history
    #                 other_past = dataloader.helper.get_past_for_agent(
    #                     agent['instance_token'],
    #                     sample_token,
    #                     seconds=2,
    #                     in_agent_frame=False,
    #                     just_xy=True
    #                 )
    #                 if len(other_past) > 0:
    #                     ax.plot(
    #                         other_past[:, 0],
    #                         other_past[:, 1],
    #                         color='yellow',  # Yellow for other agents' history
    #                         linestyle='--',
    #                         linewidth=1,
    #                         alpha=0.5
    #                     )

    #         ax.set_aspect('equal')
            
    #         # Save figure
    #         plt.savefig(
    #             os.path.join(plot_dir, f"{idx}.svg"),
    #             format='svg',
    #             bbox_inches='tight',
    #             pad_inches=0.1,
    #             transparent=True,
    #             dpi=300
    #         )
    #         plt.close(fig)

    #     print(f"Saved {min(9042, len(predictions_for_plotting))} sample images.")

    def check_driveable_trajectories_batch(self, instance_tokens, sample_tokens, trajectories):
        """
        Batch processing for driveable area compliance checking using GPU.
        """
        # Load the dataloader once to avoid repeated loading in each call
        with gin.config_scope('test'):
            dataloader = NuscenesDataset()

        # Vectorized checking of driveable areas
        results = tf.map_fn(
            lambda i: dataloader.check_driveable_trajectory_m(
                instance_tokens[i].numpy().decode(), 
                sample_tokens[i].numpy().decode(), 
                trajectories[i].numpy()
            ), 
            tf.range(len(instance_tokens)), 
            fn_output_signature=tf.bool
        )
        return results
    def check_stop_trajectories_batch(self, instance_tokens, sample_tokens, trajectories):
        """
        Batch processing for driveable area compliance checking using GPU.
        """
        # Load the dataloader once to avoid repeated loading in each call
        with gin.config_scope('test'):
            dataloader = NuscenesDataset()

        # Vectorized checking of driveable areas
        results = tf.map_fn(
            lambda i: dataloader.check_stop_trajectory_m(
                instance_tokens[i].numpy().decode(), 
                sample_tokens[i].numpy().decode(), 
                trajectories[i].numpy()
            ), 
            tf.range(len(instance_tokens)), 
            fn_output_signature=tf.bool
        )
        return results
    
    # def check_driveable_trajectory_m(self, instance_token, sample_token, trajectory):
    #     # We'll implement this method using NuscenesDataset
    #     with gin.config_scope('test'):
    #         dataloader = NuscenesDataset()
    #     return dataloader.check_driveable_trajectories_batch(instance_token, sample_token, trajectory)
        #return dataloader.check_driveable_trajectory_m(instance_token, sample_token, trajectory)


    def post_calibrate(self, lr=0.001, l2_lambda = 0.001, batch_size=128, epochs=500):
        # maybe long-term we split this into a seperate class that can be configured and called
        inputs = self.model.input
        self.model.trainable = False
        outputs = self.model(inputs, training=False)
        l2_regularizer= tf.keras.regularizers.l2(l = l2_lambda)
        # dirchilet implementation according to github for NNs using the prob scores (not logits)
        # convertion of prob scores to logarithmic scale
        def _logFunc(x):
            eps = np.finfo(float).eps
            return tf.keras.backend.log(tf.keras.backend.clip(x, eps, 1 - eps))
        x = tf.keras.layers.Lambda(_logFunc)(outputs["cls"])
 
        cls_cal = tf.keras.layers.Dense(x.shape[1], activation="softmax", 
            kernel_initializer=tf.keras.initializers.Identity(gain=1), bias_initializer="zeros", 
            kernel_regularizer=l2_regularizer, bias_regularizer=l2_regularizer, 
            name="cls")(outputs["cls"])

        outputs = {k: tf.keras.layers.Activation("linear", dtype="float32", name=k)(v) for k, v in outputs.items() if k != "cls"}
        outputs["cls"] = cls_cal
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        loss = {"cls": self.loss["cls"]}
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=loss, metrics=self._build_metrics())
        with gin.config_scope('val'):
            val_dataset = NuscenesDataset().get_dataset(shuffle=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        cbs = [tf.keras.callbacks.EarlyStopping(monitor="cls_loss", min_delta=0, patience=10, mode="auto")]
        model.fit(val_dataset, epochs=epochs, callbacks=cbs, verbose=2)
        return model
    
    def fisher_estimation(self, ckp_path):
        """
        Estimates the fisher matrix of the SNGP extractor
        """ 
        # init data
        # ckp = tf.train.Checkpoint(self.model, optimizer=self.model.optimizer)
        if self.fisher_samples > 0:
            num_samples= self.fisher_samples
            data = self.train_dataset.take(num_samples)
        else: # use complete dataset to estimate fisher matrix
            num_samples = len(self.train_dataset)
            data = self.train_dataset

        # get weight/relevance tensors
        weights = [tensor for tensor in self.model.feature_extractor.trainable_weights if "/kernel" in tensor.name]
        relevances = [tensor for tensor in self.model.feature_extractor.non_trainable_weights if "/r" in tensor.name]
        precision = [tf.zeros_like(tensor) for tensor in weights]

        # loss neeeded for fisher_type true
        def mapped_BCE(y_true, y_pred):
            return binary_crossentropy(
                *self.mapper_label(y_true, y_pred), from_logits=False
            )

        for sample in data:
            input = sample[0]
            label = sample[1]["label"]
            
            for k, v in input.items():
                input[k] = tf.expand_dims(v, axis=0)
            
            # Collect gradients.
            with tf.GradientTape() as tape:
                output = self.model(input)["label"]
                
                if self.fisher_type == "pred": # using predicted label
                    label = tf.multiply(tf.ones_like(output), tf.cast(tf.math.greater_equal(output, 0.5), output.dtype))
                    neg_log_likelihood = binary_crossentropy(label, output, from_logits=False)
                elif self.fisher_type == "true": # using true label, called "empirical" fisher -> Note: empirical fisher can be pathological
                    neg_log_likelihood = mapped_BCE(tf.expand_dims(label, axis=0), output)

            gradients = tape.gradient(neg_log_likelihood, weights)

            # If the model has converged, we can assume that the current weights
            # are the mean, and each gradient we see is a deviation. 
            precision = [var + (grad ** 2) for var, grad in zip(precision, gradients)]

        # set relevance tensors
        for i, tensor in enumerate(precision):
            relevances[i].assign(tensor / num_samples)

        # save model at same ckp
        ckp = tf.train.Checkpoint(self.model, optimizer=self.model.optimizer)
        ckp.save(ckp_path)

    def run_training(self, start_epoch=None, lr=None):
        """
        Executes a training. Overrides learning rate, if required.

        :param int start_epoch: Overrides self.start_epoch
        :param float lr: Overrides optimizer learning rate
        """
        if (start_epoch if start_epoch is not None else self.start_epoch) < self.epochs:
            if lr:
                if isinstance(self.model.optimizer.lr, LearningRateSchedule):
                    if hasattr(self.model.optimizer.lr, "end_learning_rate"):
                        tf.keras.backend.set_value(
                            self.model.optimizer.lr.end_learning_rate, lr
                        )
                    else:
                        tf.keras.backend.set_value(
                            self.model.optimizer.lr.initial_learning_rate, lr
                        )
                else:
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)

            train_dataset = self.train_dataset.batch(self.batch_size, drop_remainder=True).prefetch(
                tf.data.AUTOTUNE
            )

            if self.val_dataset:
                val_dataset = self.val_dataset.batch(self.batch_size, drop_remainder=True).prefetch(
                    tf.data.AUTOTUNE
                )
            return self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.epochs,
                initial_epoch=start_epoch
                if start_epoch is not None
                else self.start_epoch,
                callbacks=self.get_callbacks(),
                verbose=2,
            )

    def plot_predictions(self, dataset):
        pass

#@ray.remote
class TuneableTrainer(tune.Trainable):
    """
    Subclass of Ray Trainable, for model (re)-initialization

    :param dict config: Optimization config, including gin_config and load_model
    """

    def __init__(self, config, logger_creator=None):
        from shift.tf.model.models import (
            Covernet,
            CovernetDet,
            CovernetSNGP,
            CovernetVI,
        )
        from shift.util.config import paramSearch

        gin.enter_interactive_mode()
        gin.parse_config(config["gin_config"])
        del config["gin_config"]
        self.load_model = config.get("load_model",gin.config._CONFIG[('', 'shift.tf.model.training.Trainer')].get('load_model', None))
        if "load_model" in config:
            del config["load_model"]

        super().__init__(config, logger_creator)

    def setup(self, config):
        self.trainer = Trainer(
            **config, epochs=1, is_tune=True, load_model=self.load_model
        )
        self.trainer.build()
        self.config = self.trainer.hyperparameters

    def step(self):
        history = self.trainer.run_training(
            lr=self.config.get('lr', None), start_epoch=0
        )
        metrics = {k: v[-1] for k, v in history.history.items()}

        for k, v in self.config.items():
            if isinstance(v, numbers.Number):
                metrics[k] = v

        return metrics

    def load_checkpoint(self, checkpoint):
        checkpoint = Path(checkpoint)
        tmp_path = checkpoint.parent.parent / f"persistent_restore"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        shutil.copytree(checkpoint.parent, tmp_path)
        chkpt = str(list(tmp_path.glob("*.index"))[0].with_suffix(''))
        self.trainer.checkpointer.restore(chkpt)

    def save_checkpoint(self, tmp_checkpoint_dir):
        path = Path(tmp_checkpoint_dir) / f'Model'
        self.trainer.checkpointer.save(path)
        return str(Path(tmp_checkpoint_dir) / "checkpoint")
