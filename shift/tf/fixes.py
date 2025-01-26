"""
TrainableNormal distribution with non scalar scale values.

"""

from edward2.tensorflow import generated_random_variables

import edward2 as ed2
import tensorflow as tf
from keras.engine.compile_utils import (
    match_dtype_and_rank,
    get_mask,
    apply_mask,
    MetricsContainer,
)


def __call__(self, shape, dtype=None):
    if not self.built:
        self.build(shape, dtype)
    mean = self.mean
    if self.mean_constraint:
        mean = self.mean_constraint(mean)
    stddev = self.stddev
    if self.stddev_constraint:
        stddev = self.stddev_constraint(stddev)
    mean = tf.cast(mean, stddev.dtype)
    return generated_random_variables.Independent(
        generated_random_variables.Normal(loc=mean, scale=stddev).distribution,
        reinterpreted_batch_ndims=len(shape),
    )


ed2.initializers.TrainableNormal.__call__ = __call__


def update_state(self, y_true, y_pred, sample_weight=None):
    """Updates the state of per-output metrics. Modified to respect joined outputs"""
    y_true = self._conform_to_outputs(y_pred, y_true)
    sample_weight = self._conform_to_outputs(y_pred, sample_weight)
    if not self.built:
        joined = self._metrics.pop("joined") if "joined" in self._metrics else []
        self._joined = joined
        self.build(y_pred, y_true)
        for i in range(len(joined)):
            self._metrics.append(joined[i].metrics)
            self._weighted_metrics.append([])
            self._metrics_in_order.extend(joined[i].metrics)
    else:
        joined = self._joined if self._joined is not None else []

    y_pred = tf.nest.flatten(y_pred)
    y_true = tf.nest.flatten(y_true) if y_true is not None else []
    sample_weight = tf.nest.flatten(sample_weight)

    n_outputs = len(self._output_names)
    indices = []
    for i in range(len(joined)):
        y_pred.append([])
        y_true.append([])
        sample_weight.append([])
        indices.append([self._output_names.index(name) for name in joined[i].outputs])
    zip_args = (y_true, y_pred, sample_weight, self._metrics, self._weighted_metrics)
    for idx, (y_t, y_p, sw, metric_objs, weighted_metric_objs) in enumerate(zip(*zip_args)):

        for i in range(len(joined)):
            if idx in indices[i]:
                if y_t is not None:
                    y_true[n_outputs+i].append(y_t)
                y_pred[n_outputs+i].append(y_p)
                sample_weight[n_outputs+i].append(sw)

        # Ok to have no metrics for an output.
        if y_t is None or (
            all(m is None for m in metric_objs)
            and all(wm is None for wm in weighted_metric_objs)
        ):
            continue

        if idx < n_outputs:        
            y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
            mask = get_mask(y_p)
            sw = apply_mask(y_p, sw, mask)

        for metric_obj in metric_objs:
            if metric_obj is None:
                continue
            metric_obj.update_state(y_t, y_p, sample_weight=mask)

        for weighted_metric_obj in weighted_metric_objs:
            if weighted_metric_obj is None:
                continue
            weighted_metric_obj.update_state(y_t, y_p, sample_weight=sw)


MetricsContainer.update_state = update_state

# See https://github.com/tensorflow/tensorflow/issues/42872
class TensorflowFix(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TensorflowFix, self).__init__()
        self._supports_tf_logs = True
        self._backup_loss = None

    def on_train_begin(self, logs=None):
        self._backup_loss = {**self.model.loss}

    def on_train_batch_end(self, batch, logs=None):
        self.model.loss = self._backup_loss
