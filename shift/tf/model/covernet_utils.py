'''
Utility functions for CoverNet based predictors
'''
import tensorflow as tf
from shapely.geometry import LineString, Point
import numpy as np

def covernet_to_trajs(y_true, y_pred, lattice):
    """
    Maps the covernet output (class) to the trajectory represented by that class. Only applied to y_pred

    :param tf.Tensor y_true: ground truth, not used
    :param tf.Tensor y_pred: predictive class probabilities
    :param tf.Tensor lattice: array of trajectories for each class with (CLASS,STEP,POS) format
    :return: y_true, array with (CLASS,STEP,POS), ordered by the prediction probability
    :rtype: tf.Tensor, tf.Tensor
    """
    best_n = tf.argsort(y_pred, direction='DESCENDING', axis=-1)
    collected_trajs = tf.gather(
        tf.repeat(tf.expand_dims(lattice, 0), repeats=tf.shape(best_n)[0], axis=0),
        best_n,
        batch_dims=1,
    )
    return y_true, collected_trajs




def covernet_to_class(y_true, y_pred, lattice):
    """
    Converts the ground truth trajectory to the index of the closest trajectory in the covernet trajectory set

    :param tf.Tensor y_true: ground truth trajectory
    :param tf.Tensor y_pred: predictive class probabilities, not used
    :param tf.Tensor lattice: array of trajectories for each class with (CLASS,STEP,POS) format
    :return: index of the closest trajectory, y_pred
    :rtype: tf.Tensor, tf.Tensor
    """
    y_true = closest_trajectory(lattice, y_true)
    return y_true, y_pred


def covernet_to_multilabel(y_true, y_pred):
    """
    Converts the ground truth trajectories to a boolean array, defining the availability label. Assumes that y_true has the same shape as lattice and the not available trajectories have 0 everywhere.

    :param tf.Tensor y_true: ground truth trajectory
    :param tf.Tensor y_pred: predictive class probabilities, not used
    :param tf.Tensor lattice: array of trajectories for each class with (CLASS,STEP,POS) format
    :return: bool array of available trajectories, y_pred
    :rtype: tf.Tensor, tf.Tensor
    """
    y_true = tf.cast(
        tf.reduce_sum(tf.reduce_sum(y_true, axis=-1), axis=-1) > 0, tf.float32
    )
    return y_true, y_pred


def closest_trajectory(lattice, ground_truth):
    """
    Determines the closest trajectory from the CoverNet, wrt. l2 distance.

    :param tf.Tensor lattice: lattice of trajectories
    :param tf.Tensor ground_truth: single trajectory
    :return: Trajectory index
    :rtype: tf.Tensor
    """
    stacked_ground_truth = tf.cast(tf.repeat(ground_truth, repeats=lattice.shape[0], axis=1),lattice.dtype)
    result = tf.argmin(
        tf.reduce_mean(
            tf.sqrt(
                tf.reduce_sum(
                    tf.pow(tf.expand_dims(lattice, 0) - stacked_ground_truth, 2),
                    axis=-1,
                )
            ),
            axis=-1,
        ),
        axis=-1,
    )
    return result