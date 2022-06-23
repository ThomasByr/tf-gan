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
"""Tests for tfgan.features.conditioning_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tf_gan as tfgan


class ConditioningUtilsTest(tf.test.TestCase):

  def test_condition_tensor_multiple_shapes(self):
    for tensor_shape in [(4, 1), (4, 2), (4, 2, 6)]:
      for conditioning_shape in [(4, 1), (4, 8), (4, 5, 3)]:
        tfgan.features.condition_tensor(tf.zeros(tensor_shape, tf.float32),
                                        tf.zeros(conditioning_shape, tf.float32))

  def test_condition_tensor_not_fully_defined(self):
    if tf.executing_eagerly():
      return
    for conditioning_shape in [(4, 1), (4, 8), (4, 5, 3)]:
      tfgan.features.condition_tensor(tf.compat.v1.placeholder(tf.float32, (None, 5, 3)),
                                      tf.zeros(conditioning_shape, tf.float32))

  def test_condition_tensor_asserts(self):
    if tf.executing_eagerly():
      exception_type = tf.errors.InvalidArgumentError
    else:
      exception_type = ValueError
    with self.assertRaises(exception_type):
      tfgan.features.condition_tensor(tf.zeros((4, 1), tf.float32), tf.zeros((5, 1), tf.float32))

    with self.assertRaisesRegexp(ValueError, 'at least 2D'):
      tfgan.features.condition_tensor(tf.zeros((5, 2), tf.float32), tf.zeros((5), tf.float32))

  def test_condition_tensor_asserts_notfullydefined(self):
    if tf.executing_eagerly():
      return
    with self.assertRaisesRegexp(ValueError, 'Shape .* is not fully defined'):
      tfgan.features.condition_tensor(tf.compat.v1.placeholder(tf.float32, (5, None)),
                                      tf.zeros((5, 1), tf.float32))

  def test_condition_tensor_from_onehot(self):
    tfgan.features.condition_tensor_from_onehot(tf.zeros((5, 4, 1), tf.float32), tf.zeros((5, 10),
                                                                                          tf.float32))

  def test_condition_tensor_from_onehot_asserts(self):
    with self.assertRaisesRegexp(ValueError, 'Shape .* must have rank 2'):
      tfgan.features.condition_tensor_from_onehot(tf.zeros((5, 1), tf.float32), tf.zeros((5), tf.float32))

    if tf.executing_eagerly():
      exception_type = tf.errors.InvalidArgumentError
    else:
      exception_type = ValueError
    with self.assertRaises(exception_type):
      tfgan.features.condition_tensor_from_onehot(tf.zeros((5, 1), tf.float32), tf.zeros((4, 6), tf.float32))

  def test_condition_tensor_from_onehot_asserts_notfullydefined(self):
    if tf.executing_eagerly():
      return
    with self.assertRaisesRegexp(ValueError, 'Shape .* is not fully defined'):
      tfgan.features.condition_tensor_from_onehot(tf.zeros((5, 1), tf.float32),
                                                  tf.compat.v1.placeholder(tf.float32, (5, None)))


if __name__ == '__main__':
  tf.test.main()
