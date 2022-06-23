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
"""Tests for mnist.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow.compat.v1 as tf

from tf_gan.examples.mnist import train_lib

mock = tf.test.mock

BATCH_SIZE = 5


def _new_data(*args, **kwargs):
  del args, kwargs
  # Tensors need to be created in the same graph, so generate them at the call
  # site.
  # Note: Make sure batch size matches hparams.
  imgs = tf.zeros([BATCH_SIZE, 28, 28, 1], dtype=tf.float32)
  labels = tf.one_hot([0] * BATCH_SIZE, depth=10)
  return (imgs, labels)


class TrainTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TrainTest, self).setUp()
    self.hparams = train_lib.HParams(batch_size=BATCH_SIZE,
                                     train_log_dir=self.get_temp_dir(),
                                     max_number_of_steps=1,
                                     gan_type='unconditional',
                                     grid_size=1,
                                     noise_dims=64)

  @mock.patch.object(train_lib.data_provider, 'provide_data', new=_new_data)
  def test_run_one_train_step(self):
    if tf.executing_eagerly():
      # `tfgan.gan_model` doesn't work when executing eagerly.
      return
    train_lib.train(self.hparams)

  @parameterized.parameters(
      {'gan_type': 'unconditional'},
      {'gan_type': 'conditional'},
      {'gan_type': 'infogan'},
  )
  @mock.patch.object(train_lib.data_provider, 'provide_data', new=_new_data)
  def test_build_graph(self, gan_type):
    if tf.executing_eagerly():
      # `tfgan.gan_model` doesn't work when executing eagerly.
      return
    hparams = self.hparams._replace(max_number_of_steps=0, gan_type=gan_type)
    train_lib.train(hparams)


if __name__ == '__main__':
  tf.test.main()
