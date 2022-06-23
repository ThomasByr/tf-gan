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
"""Tests for tfgan.features.random_tensor_pool."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tf_gan as tfgan


class TensorPoolTest(tf.test.TestCase):

  def test_pool_unknown_input_shape(self):
    """Checks that `input_value` can have unknown shape."""
    if tf.executing_eagerly():
      # Placeholders don't work in eager execution mode.
      return
    input_value = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None, 3])
    output_value = tfgan.features.tensor_pool(input_value, pool_size=10)
    self.assertEqual(output_value.shape.as_list(), [None, None, 3])

    with self.cached_session() as sess:
      for i in range(10):
        sess.run(output_value, {input_value: [[[i] * 3]]})
        sess.run(output_value, {input_value: [[[i] * 3] * 2]})
        sess.run(output_value, {input_value: [[[i] * 3] * 5] * 2})

  def test_pool_sequence(self):
    """Checks that values are pooled and returned maximally twice."""
    if tf.executing_eagerly():
      # Placeholders aren't supported in eager execution.
      return
    input_value = tf.compat.v1.placeholder(dtype=tf.int32, shape=[])
    output_value = tfgan.features.tensor_pool(input_value, pool_size=10)
    self.assertEqual(output_value.shape.as_list(), [])

    with self.cached_session() as session:
      outs = []
      for i in range(50):
        out = session.run(output_value, {input_value: i})
        outs.append(out)
        self.assertLessEqual(out, i)

      _, counts = np.unique(outs, return_counts=True)
      # Check that each value is returned maximally twice.
      self.assertTrue((counts <= 2).all())

  def test_never_pool(self):
    """Checks that setting `pooling_probability` to zero works."""
    if tf.executing_eagerly():
      # Placeholders don't work in eager execution mode.
      return
    input_value = tf.compat.v1.placeholder(dtype=tf.int32, shape=[])
    output_value = tfgan.features.tensor_pool(input_value, pool_size=10, pooling_probability=0.0)
    self.assertEqual(output_value.shape.as_list(), [])

    with self.cached_session() as session:
      for i in range(50):
        out = session.run(output_value, {input_value: i})
        self.assertEqual(out, i)

  def test_pooling_probability(self):
    """Checks that `pooling_probability` works."""
    if tf.executing_eagerly():
      # Placeholders don't work in eager execution mode.
      return
    input_value = tf.compat.v1.placeholder(dtype=tf.int32, shape=[])
    pool_size = 10
    pooling_probability = 0.2
    output_value = tfgan.features.tensor_pool(input_value,
                                              pool_size=pool_size,
                                              pooling_probability=pooling_probability)
    self.assertEqual(output_value.shape.as_list(), [])

    with self.cached_session() as session:
      not_pooled = 0
      total = 1000
      for i in range(total):
        out = session.run(output_value, {input_value: i})
        if out == i:
          not_pooled += 1
      self.assertAllClose((not_pooled-pool_size) / (total-pool_size), 1 - pooling_probability, atol=0.03)

  def test_input_values_tuple(self):
    """Checks that `input_values` can be a tuple."""
    if tf.executing_eagerly():
      # Placeholders don't work in eager execution mode.
      return
    input_values = (tf.compat.v1.placeholder(dtype=tf.int32,
                                             shape=[]), tf.compat.v1.placeholder(dtype=tf.int32, shape=[]))
    output_values = tfgan.features.tensor_pool(input_values, pool_size=3)
    self.assertEqual(len(output_values), len(input_values))
    for output_value in output_values:
      self.assertEqual(output_value.shape.as_list(), [])

    with self.cached_session() as session:
      for i in range(10):
        outs = session.run(output_values, {input_values[0]: i, input_values[1]: i + 1})
        self.assertEqual(len(outs), len(input_values))
        self.assertEqual(outs[1] - outs[0], 1)

  def test_pool_preserves_shape(self):
    t = tf.constant(1)
    input_values = [[t, t, t], (t, t), t]
    output_values = tfgan.features.tensor_pool(input_values, pool_size=5)
    # Overall shape.
    self.assertIsInstance(output_values, list)
    self.assertEqual(3, len(output_values))
    # Shape of first element.
    self.assertIsInstance(output_values[0], list)
    self.assertEqual(3, len(output_values[0]))
    # Shape of second element.
    self.assertIsInstance(output_values[1], tuple)
    self.assertEqual(2, len(output_values[1]))
    # Shape of third element.
    self.assertIsInstance(output_values[2], tf.Tensor)


if __name__ == '__main__':
  tf.test.main()
