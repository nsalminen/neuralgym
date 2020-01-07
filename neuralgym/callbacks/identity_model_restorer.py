"""keras_model_restorer"""

import tensorflow as tf

from . import CallbackLoc, OnceCallback
from ..utils.logger import callback_log


class IdentityModelRestorer(OnceCallback):
    """Restore identity model from temp file

    Args:
        model_weights: list of variables of identity model graph.
        tf_checkpoint_path (str): path of checkpoint.
    """

    def __init__(self, model_weights, tf_checkpoint_path):
        super().__init__(CallbackLoc.train_start)
        self._model_weights = model_weights
        self._tf_checkpoint_path = tf_checkpoint_path

    def run(self, sess):
        tf.train.Saver(self._model_weights).restore(sess, self._tf_checkpoint_path)
        callback_log('Trigger identity model weights loader: weights restored from %s.' % self._tf_checkpoint_path)
