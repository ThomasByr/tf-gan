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
"""Tests for tfgan.features.clip_weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
import tf_gan as tfgan


class ClipWeightsTest(tf.test.TestCase):
  """Tests for `discriminator_weight_clip`."""

  def setUp(self):
    super(ClipWeightsTest, self).setUp()
    self.variables = [tf.Variable(2.0)]
    self.tuple = collections.namedtuple('VarTuple', ['discriminator_variables'])(self.variables)

  def _test_weight_clipping_helper(self, use_tuple):
    loss = self.variables[0]
    opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)
    if use_tuple:
      opt_clip = tfgan.features.clip_variables(opt, self.variables, 0.1)
    else:
      opt_clip = tfgan.features.clip_discriminator_weights(opt, self.tuple, 0.1)

    train_op1 = opt.minimize(loss, var_list=self.variables)
    train_op2 = opt_clip.minimize(loss, var_list=self.variables)

    with self.cached_session(use_gpu=True) as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertEqual(2.0, sess.run(self.variables[0]))
      sess.run(train_op1)
      self.assertLess(0.1, sess.run(self.variables[0]))

    with self.cached_session(use_gpu=True) as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self.assertEqual(2.0, sess.run(self.variables[0]))
      sess.run(train_op2)
      self.assertNear(0.1, sess.run(self.variables[0]), 1e-7)

  def test_weight_clipping_argsonly(self):
    if tf.executing_eagerly():
      # Optimizers work differently in eager.
      return
    self._test_weight_clipping_helper(False)

  def test_weight_clipping_ganmodel(self):
    if tf.executing_eagerly():
      # Optimizers work differently in eager.
      return
    self._test_weight_clipping_helper(True)

  def _test_incorrect_weight_clip_value_helper(self, use_tuple):
    opt = tf.compat.v1.train.GradientDescentOptimizer(1.0)

    if use_tuple:
      with self.assertRaisesRegexp(ValueError, 'must be positive'):
        tfgan.features.clip_discriminator_weights(opt, self.tuple, weight_clip=-1)
    else:
      with self.assertRaisesRegexp(ValueError, 'must be positive'):
        tfgan.features.clip_variables(opt, self.variables, weight_clip=-1)

  def test_incorrect_weight_clip_value_argsonly(self):
    self._test_incorrect_weight_clip_value_helper(False)

  def test_incorrect_weight_clip_value_tuple(self):
    self._test_incorrect_weight_clip_value_helper(True)


if __name__ == '__main__':
  tf.test.main()
