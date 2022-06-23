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
"""Evaluates an InfoGAN TFGAN trained MNIST model.

The image visualizations, as in https://arxiv.org/abs/1606.03657, show the
effect of varying a specific latent variable on the image. Each visualization
focuses on one of the three structured variables. Columns have two of the three
variables fixed, while the third one is varied. Different rows have different
random samples from the remaining latents.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow.compat.v1 as tf
import tf_gan as tfgan

from tf_gan.examples import evaluation_helper as evaluation
from tf_gan.examples.mnist import data_provider, networks, util

HParams = collections.namedtuple('HParams', [
    'checkpoint_dir', 'eval_dir', 'noise_samples', 'unstructured_noise_dims', 'continuous_noise_dims',
    'max_number_of_evaluations', 'write_to_disk'
])

CAT_SAMPLE_POINTS = np.arange(0, 10)
CONT_SAMPLE_POINTS = np.linspace(-2.0, 2.0, 10)


def evaluate(hparams, run_eval_loop=True):
  """Runs an evaluation loop.

  Args:
    hparams: An HParams instance containing the eval hyperparameters.
    run_eval_loop: Whether to run the full eval loop. Set to False for testing.
  """
  with tf.name_scope('inputs'):
    noise_args = (hparams.noise_samples, CAT_SAMPLE_POINTS, CONT_SAMPLE_POINTS,
                  hparams.unstructured_noise_dims, hparams.continuous_noise_dims)
    # Use fixed noise vectors to illustrate the effect of each dimension.
    display_noise1 = util.get_eval_noise_categorical(*noise_args)
    display_noise2 = util.get_eval_noise_continuous_dim1(*noise_args)
    display_noise3 = util.get_eval_noise_continuous_dim2(*noise_args)
    _validate_noises([display_noise1, display_noise2, display_noise3])

  # Visualize the effect of each structured noise dimension on the generated
  # image.
  def generator_fn(inputs):
    return networks.infogan_generator(inputs, len(CAT_SAMPLE_POINTS), is_training=False)

  with tf.variable_scope('Generator') as genscope:  # Same scope as in training.
    categorical_images = generator_fn(display_noise1)
  reshaped_categorical_img = tfgan.eval.image_reshaper(categorical_images, num_cols=len(CAT_SAMPLE_POINTS))
  tf.summary.image('categorical', reshaped_categorical_img, max_outputs=1)

  with tf.variable_scope(genscope, reuse=True):
    continuous1_images = generator_fn(display_noise2)
  reshaped_continuous1_img = tfgan.eval.image_reshaper(continuous1_images, num_cols=len(CONT_SAMPLE_POINTS))
  tf.summary.image('continuous1', reshaped_continuous1_img, max_outputs=1)

  with tf.variable_scope(genscope, reuse=True):
    continuous2_images = generator_fn(display_noise3)
  reshaped_continuous2_img = tfgan.eval.image_reshaper(continuous2_images, num_cols=len(CONT_SAMPLE_POINTS))
  tf.summary.image('continuous2', reshaped_continuous2_img, max_outputs=1)

  # Evaluate image quality.
  all_images = tf.concat([categorical_images, continuous1_images, continuous2_images], 0)
  tf.summary.scalar('MNIST_Classifier_score', util.mnist_score(all_images))

  # Write images to disk.
  image_write_ops = []
  if hparams.write_to_disk:
    image_write_ops.append(
        _get_write_image_ops(hparams.eval_dir, 'categorical_infogan.png', reshaped_categorical_img[0]))
    image_write_ops.append(
        _get_write_image_ops(hparams.eval_dir, 'continuous1_infogan.png', reshaped_continuous1_img[0]))
    image_write_ops.append(
        _get_write_image_ops(hparams.eval_dir, 'continuous2_infogan.png', reshaped_continuous2_img[0]))

  # For unit testing, use `run_eval_loop=False`.
  if not run_eval_loop:
    return
  evaluation.evaluate_repeatedly(
      hparams.checkpoint_dir,
      hooks=[evaluation.SummaryAtEndHook(hparams.eval_dir),
             evaluation.StopAfterNEvalsHook(1)],
      eval_ops=image_write_ops,
      max_number_of_evaluations=hparams.max_number_of_evaluations)


def _validate_noises(noises):
  """Sanity check on constructed noise tensors.

  Args:
    noises: List of 3-tuples of noise vectors.
  """
  assert isinstance(noises, (list, tuple))
  for noise_l in noises:
    assert len(noise_l) == 3
    assert isinstance(noise_l[0], np.ndarray)
    batch_dim = noise_l[0].shape[0]
    for i, noise in enumerate(noise_l):
      assert isinstance(noise, np.ndarray)
      # Check that batch dimensions are all the same.
      assert noise.shape[0] == batch_dim

      # Check that shapes for corresponding noises are the same.
      assert noise.shape == noises[0][i].shape


def _get_write_image_ops(eval_dir, filename, images):
  """Create Ops that write images to disk."""
  return tf.io.write_file('%s/%s' % (eval_dir, filename),
                          tf.image.encode_png(data_provider.float_image_to_uint8(images)))
