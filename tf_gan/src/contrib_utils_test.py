#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# TF-GAN
#
# Copyright (c) 2022, ThomasByr.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# - Neither the name of the tf-gan authors nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Tests for tf_gan.python.contrib_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
import tensorflow as tf
from tf_gan.src import contrib_utils


def batchnorm_classifier(inputs):
  bn_layer = tf.keras.layers.BatchNormalization(momentum=0.1)
  inputs = bn_layer(inputs, training=True)
  assert bn_layer.updates
  for update in bn_layer.updates:
    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, update)
  return tf.compat.v1.layers.dense(inputs, 1, activation=tf.sigmoid)


class CreateTrainOpTest(tf.test.TestCase, absltest.TestCase):

  def setUp(self):
    super(CreateTrainOpTest, self).setUp()
    np.random.seed(0)

    # Create an easy training set:
    self._inputs = np.random.rand(16, 4).astype(np.float32)
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

  def testTrainOpInCollection(self):
    with tf.Graph().as_default():
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      self.assertNotEmpty(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS))
      loss = tf.compat.v1.losses.log_loss(tf_labels, tf_predictions)
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = contrib_utils.create_train_op(loss, optimizer)

      # Make sure the training op was recorded in the proper collection
      self.assertIn(train_op, tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAIN_OP))

  def testUseUpdateOps(self):
    with tf.Graph().as_default():
      tf.compat.v1.random.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      expected_mean = np.mean(self._inputs, axis=(0))
      expected_var = np.var(self._inputs, axis=(0))

      tf_predictions = batchnorm_classifier(tf_inputs)
      self.assertNotEmpty(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS))
      loss = tf.compat.v1.losses.log_loss(tf_labels, tf_predictions)
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = contrib_utils.create_train_op(loss, optimizer)

      moving_mean = contrib_utils.get_variables_by_name('moving_mean')[0]
      moving_variance = contrib_utils.get_variables_by_name('moving_variance')[0]

      with self.cached_session() as sess:
        # Initialize all variables
        sess.run(tf.compat.v1.global_variables_initializer())
        mean, variance = sess.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)

        for _ in range(10):
          sess.run(train_op)

        mean = sess.run(moving_mean)
        variance = sess.run(moving_variance)
        # After 10 updates with decay 0.1 moving_mean == expected_mean and
        # moving_variance == expected_var.
        self.assertAllClose(mean, expected_mean)
        self.assertAllClose(variance, expected_var)

  def testEmptyUpdateOps(self):
    with tf.Graph().as_default():
      tf.compat.v1.random.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      self.assertNotEmpty(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS))
      loss = tf.compat.v1.losses.log_loss(tf_labels, tf_predictions)
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = contrib_utils.create_train_op(loss, optimizer, update_ops=[])

      moving_mean = contrib_utils.get_variables_by_name('moving_mean')[0]
      moving_variance = contrib_utils.get_variables_by_name('moving_variance')[0]

      with self.cached_session() as sess:
        # Initialize all variables
        sess.run(tf.compat.v1.global_variables_initializer())
        mean, variance = sess.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)

        for _ in range(10):
          sess.run(train_op)

        mean = sess.run(moving_mean)
        variance = sess.run(moving_variance)

        # Since we skip update_ops the moving_vars are not updated.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)

  def testGlobalStepIsIncrementedByDefault(self):
    with tf.Graph().as_default():
      tf.compat.v1.random.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      self.assertNotEmpty(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS))
      loss = tf.compat.v1.losses.log_loss(tf_labels, tf_predictions)
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = contrib_utils.create_train_op(loss, optimizer)

      global_step = tf.compat.v1.train.get_or_create_global_step()

      with self.cached_session() as sess:
        # Initialize all variables
        sess.run(tf.compat.v1.global_variables_initializer())

        for _ in range(10):
          sess.run(train_op)

        # After 10 updates global_step should be 10.
        self.assertAllClose(sess.run(global_step), 10)

  def testGlobalStepNotIncrementedWhenSetToNone(self):
    with tf.Graph().as_default():
      tf.compat.v1.random.set_random_seed(0)
      tf_inputs = tf.constant(self._inputs, dtype=tf.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.float32)

      tf_predictions = batchnorm_classifier(tf_inputs)
      self.assertNotEmpty(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS))
      loss = tf.compat.v1.losses.log_loss(tf_labels, tf_predictions)
      optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = contrib_utils.create_train_op(loss, optimizer, global_step=None)

      global_step = tf.compat.v1.train.get_or_create_global_step()

      with self.cached_session() as sess:
        # Initialize all variables
        sess.run(tf.compat.v1.global_variables_initializer())

        for _ in range(10):
          sess.run(train_op)

        # Since train_op don't use global_step it shouldn't change.
        self.assertAllClose(sess.run(global_step), 0)


if __name__ == '__main__':
  absltest.main()
