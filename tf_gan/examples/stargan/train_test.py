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
"""Tests for stargan.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan

import train_lib

mock = tf.test.mock


def _test_generator(input_images, _):
  """Simple generator function."""
  return input_images * tf.get_variable('dummy_g', initializer=2.0)


def _test_discriminator(inputs, num_domains):
  """Differentiable dummy discriminator for StarGAN."""
  hidden = tf.layers.flatten(inputs)
  output_src = tf.reduce_mean(input_tensor=hidden, axis=1)
  output_cls = tf.layers.dense(inputs=hidden, units=num_domains)

  return output_src, output_cls


train_lib.network.generator = _test_generator
train_lib.network.discriminator = _test_discriminator


class TrainTest(tf.test.TestCase):

  def setUp(self):
    super(TrainTest, self).setUp()

    # Force the TF lazy loading to kick in before mocking this out below.
    _ = tf.train.get_or_create_global_step()

    self.hparams = train_lib.HParams(batch_size=6,
                                     patch_size=128,
                                     train_log_dir='/tmp/tfgan_logdir/stargan/',
                                     generator_lr=1e-4,
                                     discriminator_lr=1e-4,
                                     max_number_of_steps=1000000,
                                     adam_beta1=0.5,
                                     adam_beta2=0.999,
                                     gen_disc_step_ratio=0.2,
                                     tf_master='',
                                     ps_replicas=0,
                                     task=0)

  def test_define_model(self):
    if tf.executing_eagerly():
      # `tfgan.stargan_model` doesn't work when executing eagerly.
      return
    hparams = self.hparams._replace(batch_size=2)
    images_shape = [hparams.batch_size, 4, 4, 3]
    images_np = np.zeros(shape=images_shape)
    images = tf.constant(images_np, dtype=tf.float32)
    labels = tf.one_hot([0] * hparams.batch_size, 2)

    model = train_lib._define_model(images, labels)
    self.assertIsInstance(model, tfgan.StarGANModel)
    self.assertShapeEqual(images_np, model.generated_data)
    self.assertShapeEqual(images_np, model.reconstructed_data)
    self.assertTrue(isinstance(model.discriminator_variables, list))
    self.assertTrue(isinstance(model.generator_variables, list))
    self.assertIsInstance(model.discriminator_scope, tf.VariableScope)
    self.assertTrue(model.generator_scope, tf.VariableScope)
    self.assertTrue(callable(model.discriminator_fn))
    self.assertTrue(callable(model.generator_fn))

  @mock.patch.object(tf.train, 'get_or_create_global_step', autospec=True)
  def test_get_lr(self, mock_get_or_create_global_step):
    if tf.executing_eagerly():
      return
    max_number_of_steps = 10
    base_lr = 0.01
    with self.cached_session(use_gpu=True) as sess:
      mock_get_or_create_global_step.return_value = tf.constant(2)
      lr_step2 = sess.run(train_lib._get_lr(base_lr, max_number_of_steps))
      mock_get_or_create_global_step.return_value = tf.constant(9)
      lr_step9 = sess.run(train_lib._get_lr(base_lr, max_number_of_steps))

    self.assertAlmostEqual(base_lr, lr_step2)
    self.assertAlmostEqual(base_lr * 0.2, lr_step9)

  def test_define_train_ops(self):
    if tf.executing_eagerly():
      # `tfgan.stargan_model` doesn't work when executing eagerly.
      return
    hparams = self.hparams._replace(batch_size=2, generator_lr=0.1, discriminator_lr=0.01)

    images_shape = [hparams.batch_size, 4, 4, 3]
    images = tf.zeros(images_shape, dtype=tf.float32)
    labels = tf.one_hot([0] * hparams.batch_size, 2)

    model = train_lib._define_model(images, labels)
    loss = tfgan.stargan_loss(model)
    train_ops = train_lib._define_train_ops(model, loss, hparams.generator_lr, hparams.discriminator_lr,
                                            hparams.adam_beta1, hparams.adam_beta2,
                                            hparams.max_number_of_steps)

    self.assertIsInstance(train_ops, tfgan.GANTrainOps)

  def test_get_train_step(self):
    gen_disc_step_ratio = 0.5
    train_steps = train_lib._define_train_step(gen_disc_step_ratio)
    self.assertEqual(1, train_steps.generator_train_steps)
    self.assertEqual(2, train_steps.discriminator_train_steps)

    gen_disc_step_ratio = 3
    train_steps = train_lib._define_train_step(gen_disc_step_ratio)
    self.assertEqual(3, train_steps.generator_train_steps)
    self.assertEqual(1, train_steps.discriminator_train_steps)

  @mock.patch.object(train_lib.data_provider, 'provide_data', autospec=True)
  def test_main(self, mock_provide_data):
    if tf.executing_eagerly():
      # `tfgan.stargan_model` doesn't work when executing eagerly.
      return
    hparams = self.hparams._replace(batch_size=2, max_number_of_steps=10)
    num_domains = 3

    # Construct mock inputs.
    images_shape = [hparams.batch_size, hparams.patch_size, hparams.patch_size, 3]
    img_list = [tf.zeros(images_shape)] * num_domains
    lbl_list = [tf.one_hot([0] * hparams.batch_size, num_domains)] * num_domains
    mock_provide_data.return_value = (img_list, lbl_list)

    train_lib.train(hparams)


if __name__ == '__main__':
  tf.test.main()
