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
"""Trains a generator on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tf_gan as tfgan

from tf_gan.examples.mnist import data_provider, networks, util

HParams = collections.namedtuple(
    'HParams', ['batch_size', 'train_log_dir', 'max_number_of_steps', 'gan_type', 'grid_size', 'noise_dims'])


def _learning_rate(gan_type):
  # First is generator learning rate, second is discriminator learning rate.
  return {
      'unconditional': (1e-3, 1e-4),
      'conditional': (1e-5, 1e-4),
      'infogan': (1e-4, 1e-4),
  }[gan_type]


def train(hparams):
  """Trains an MNIST GAN.

  Args:
    hparams: An HParams instance containing the hyperparameters for training.
  """
  if not tf.io.gfile.exists(hparams.train_log_dir):
    tf.io.gfile.makedirs(hparams.train_log_dir)

  # Force all input processing onto CPU in order to reserve the GPU for
  # the forward inference and back-propagation.
  with tf.name_scope('inputs'), tf.device('/cpu:0'):
    images, one_hot_labels = data_provider.provide_data('train', hparams.batch_size, num_parallel_calls=4)

  # Define the GANModel tuple. Optionally, condition the GAN on the label or
  # use an InfoGAN to learn a latent representation.
  if hparams.gan_type == 'unconditional':
    gan_model = tfgan.gan_model(generator_fn=networks.unconditional_generator,
                                discriminator_fn=networks.unconditional_discriminator,
                                real_data=images,
                                generator_inputs=tf.random.normal([hparams.batch_size, hparams.noise_dims]))
  elif hparams.gan_type == 'conditional':
    noise = tf.random.normal([hparams.batch_size, hparams.noise_dims])
    gan_model = tfgan.gan_model(generator_fn=networks.conditional_generator,
                                discriminator_fn=networks.conditional_discriminator,
                                real_data=images,
                                generator_inputs=(noise, one_hot_labels))
  elif hparams.gan_type == 'infogan':
    cat_dim, cont_dim = 10, 2
    generator_fn = functools.partial(networks.infogan_generator, categorical_dim=cat_dim)
    discriminator_fn = functools.partial(networks.infogan_discriminator,
                                         categorical_dim=cat_dim,
                                         continuous_dim=cont_dim)
    unstructured_inputs, structured_inputs = util.get_infogan_noise(hparams.batch_size, cat_dim, cont_dim,
                                                                    hparams.noise_dims)
    gan_model = tfgan.infogan_model(generator_fn=generator_fn,
                                    discriminator_fn=discriminator_fn,
                                    real_data=images,
                                    unstructured_generator_inputs=unstructured_inputs,
                                    structured_generator_inputs=structured_inputs)
  tfgan.eval.add_gan_model_image_summaries(gan_model, hparams.grid_size)

  # Get the GANLoss tuple. You can pass a custom function, use one of the
  # already-implemented losses from the losses library, or use the defaults.
  with tf.name_scope('loss'):
    if hparams.gan_type == 'infogan':
      gan_loss = tfgan.gan_loss(gan_model,
                                generator_loss_fn=tfgan.losses.modified_generator_loss,
                                discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
                                mutual_information_penalty_weight=1.0,
                                add_summaries=True)
    else:
      gan_loss = tfgan.gan_loss(gan_model, add_summaries=True)
    tfgan.eval.add_regularization_loss_summaries(gan_model)

  # Get the GANTrain ops using custom optimizers.
  with tf.name_scope('train'):
    gen_lr, dis_lr = _learning_rate(hparams.gan_type)
    train_ops = tfgan.gan_train_ops(gan_model,
                                    gan_loss,
                                    generator_optimizer=tf.train.AdamOptimizer(gen_lr, 0.5),
                                    discriminator_optimizer=tf.train.AdamOptimizer(dis_lr, 0.5),
                                    summarize_gradients=True,
                                    aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

  # Run the alternating training loop. Skip it if no steps should be taken
  # (used for graph construction tests).
  status_message = tf.strings.join(
      ['Starting train step: ', tf.as_string(tf.train.get_or_create_global_step())], name='status_message')
  if hparams.max_number_of_steps == 0:
    return
  tfgan.gan_train(train_ops,
                  hooks=[
                      tf_estimator.StopAtStepHook(num_steps=hparams.max_number_of_steps),
                      tf_estimator.LoggingTensorHook([status_message], every_n_iter=10)
                  ],
                  logdir=hparams.train_log_dir,
                  get_hooks_fn=tfgan.get_joint_train_hooks(),
                  save_checkpoint_secs=60)
