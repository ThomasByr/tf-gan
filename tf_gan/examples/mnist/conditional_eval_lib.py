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
"""Evaluates a conditional TF-GAN trained MNIST model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v1 as tf
import tf_gan as tfgan

from tf_gan.examples import evaluation_helper as evaluation
import data_provider
import networks
import util

NUM_CLASSES = 10

HParams = collections.namedtuple('HParams', [
    'checkpoint_dir', 'eval_dir', 'num_images_per_class', 'noise_dims', 'max_number_of_evaluations',
    'write_to_disk'
])


def evaluate(hparams, run_eval_loop=True):
  """Runs an evaluation loop.

  Args:
    hparams: An HParams instance containing the eval hyperparameters.
    run_eval_loop: Whether to run the full eval loop. Set to False for testing.
  """
  with tf.name_scope('inputs'):
    noise, one_hot_labels = _get_generator_inputs(hparams.num_images_per_class, NUM_CLASSES,
                                                  hparams.noise_dims)

  # Generate images.
  with tf.variable_scope('Generator'):  # Same scope as in train job.
    images = networks.conditional_generator((noise, one_hot_labels), is_training=False)

  # Visualize images.
  reshaped_img = tfgan.eval.image_reshaper(images, num_cols=hparams.num_images_per_class)
  tf.summary.image('generated_images', reshaped_img, max_outputs=1)

  # Calculate evaluation metrics.
  tf.summary.scalar('MNIST_Classifier_score', util.mnist_score(images))
  tf.summary.scalar('MNIST_Cross_entropy', util.mnist_cross_entropy(images, one_hot_labels))

  # Write images to disk.
  image_write_ops = None
  if hparams.write_to_disk:
    image_write_ops = tf.io.write_file(
        '%s/%s' % (hparams.eval_dir, 'conditional_gan.png'),
        tf.image.encode_png(data_provider.float_image_to_uint8(reshaped_img[0])))

  # For unit testing, use `run_eval_loop=False`.
  if not run_eval_loop:
    return
  evaluation.evaluate_repeatedly(
      hparams.checkpoint_dir,
      hooks=[evaluation.SummaryAtEndHook(hparams.eval_dir),
             evaluation.StopAfterNEvalsHook(1)],
      eval_ops=image_write_ops,
      max_number_of_evaluations=hparams.max_number_of_evaluations)


def _get_generator_inputs(num_images_per_class, num_classes, noise_dims):
  """Return generator inputs for evaluation."""
  # Since we want a grid of numbers for the conditional generator, manually
  # construct the desired class labels.
  num_images_generated = num_images_per_class * num_classes
  noise = tf.random.normal([num_images_generated, noise_dims])
  # pylint:disable=g-complex-comprehension
  labels = [lbl for lbl in range(num_classes) for _ in range(num_images_per_class)]
  one_hot_labels = tf.one_hot(tf.constant(labels), num_classes)
  return noise, one_hot_labels
