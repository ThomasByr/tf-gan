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
"""Trains a generator on CIFAR data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tf_gan as tfgan

from tf_gan.examples.cifar10 import data_provider, networks

HParams = collections.namedtuple('HParams', [
    'batch_size',
    'max_number_of_steps',
    'generator_lr',
    'discriminator_lr',
    'master',
    'train_log_dir',
    'ps_replicas',
    'task',
])


def train(hparams):
  """Trains a CIFAR10 GAN.

  Args:
    hparams: An HParams instance containing the hyperparameters for training.
  """
  if not tf.io.gfile.exists(hparams.train_log_dir):
    tf.io.gfile.makedirs(hparams.train_log_dir)

  with tf.device(tf.train.replica_device_setter(hparams.ps_replicas)):
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.name_scope('inputs'):
      with tf.device('/cpu:0'):
        images, _ = data_provider.provide_data('train', hparams.batch_size, num_parallel_calls=4)

    # Define the GANModel tuple.
    generator_fn = networks.generator
    discriminator_fn = networks.discriminator
    generator_inputs = tf.random.normal([hparams.batch_size, 64])
    gan_model = tfgan.gan_model(generator_fn,
                                discriminator_fn,
                                real_data=images,
                                generator_inputs=generator_inputs)
    tfgan.eval.add_gan_model_image_summaries(gan_model)

    # Get the GANLoss tuple. Use the selected GAN loss functions.
    with tf.name_scope('loss'):
      gan_loss = tfgan.gan_loss(gan_model, gradient_penalty_weight=1.0, add_summaries=True)

    # Get the GANTrain ops using the custom optimizers and optional
    # discriminator weight clipping.
    with tf.name_scope('train'):
      gen_opt, dis_opt = _get_optimizers(hparams)
      train_ops = tfgan.gan_train_ops(gan_model,
                                      gan_loss,
                                      generator_optimizer=gen_opt,
                                      discriminator_optimizer=dis_opt,
                                      summarize_gradients=True)

    # Run the alternating training loop. Skip it if no steps should be taken
    # (used for graph construction tests).
    status_message = tf.strings.join(
        ['Starting train step: ', tf.as_string(tf.train.get_or_create_global_step())], name='status_message')
    if hparams.max_number_of_steps == 0:
      return
    tfgan.gan_train(train_ops,
                    hooks=([
                        tf_estimator.StopAtStepHook(num_steps=hparams.max_number_of_steps),
                        tf_estimator.LoggingTensorHook([status_message], every_n_iter=10)
                    ]),
                    logdir=hparams.train_log_dir,
                    master=hparams.master,
                    is_chief=hparams.task == 0)


def _get_optimizers(hparams):
  """Get optimizers that are optionally synchronous."""
  gen_opt = tf.train.AdamOptimizer(hparams.generator_lr, 0.5)
  dis_opt = tf.train.AdamOptimizer(hparams.discriminator_lr, 0.5)

  return gen_opt, dis_opt
