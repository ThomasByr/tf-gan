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
"""Evaluates a TF-GAN trained CIFAR model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v1 as tf
import tf_gan as tfgan
from tf_gan.examples import evaluation_helper as evaluation

from tf_gan.examples.cifar10 import data_provider, networks, util

HParams = collections.namedtuple('HParams', [
    'master', 'checkpoint_dir', 'eval_dir', 'num_images_generated', 'num_inception_images',
    'eval_real_images', 'eval_frechet_inception_distance', 'max_number_of_evaluations', 'write_to_disk'
])


def evaluate(hparams, run_eval_loop=True):
  """Runs an evaluation loop.

  Args:
    hparams: An HParams instance containing the eval hyperparameters.
    run_eval_loop: Whether to run the full eval loop. Set to False for testing.
  """
  # Fetch and generate images to run through Inception.
  with tf.name_scope('inputs'):
    real_data, _ = data_provider.provide_data('test', hparams.num_images_generated, shuffle=False)
    generated_data = _get_generated_data(hparams.num_images_generated)

  # Compute Frechet Inception Distance.
  if hparams.eval_frechet_inception_distance:
    fid = util.get_frechet_inception_distance(real_data, generated_data, hparams.num_images_generated,
                                              hparams.num_inception_images)
    tf.summary.scalar('frechet_inception_distance', fid)

  # Compute normal Inception scores.
  if hparams.eval_real_images:
    inc_score = util.get_inception_scores(real_data, hparams.num_images_generated,
                                          hparams.num_inception_images)
  else:
    inc_score = util.get_inception_scores(generated_data, hparams.num_images_generated,
                                          hparams.num_inception_images)
  tf.summary.scalar('inception_score', inc_score)

  # Create ops that write images to disk.
  image_write_ops = None
  if hparams.num_images_generated >= 100 and hparams.write_to_disk:
    reshaped_imgs = tfgan.eval.image_reshaper(generated_data[:100], num_cols=10)
    uint8_images = data_provider.float_image_to_uint8(reshaped_imgs)
    image_write_ops = tf.io.write_file('%s/%s' % (hparams.eval_dir, 'unconditional_cifar10.png'),
                                       tf.image.encode_png(uint8_images[0]))

  # For unit testing, use `run_eval_loop=False`.
  if not run_eval_loop:
    return
  evaluation.evaluate_repeatedly(
      hparams.checkpoint_dir,
      master=hparams.master,
      hooks=[evaluation.SummaryAtEndHook(hparams.eval_dir),
             evaluation.StopAfterNEvalsHook(1)],
      eval_ops=image_write_ops,
      max_number_of_evaluations=hparams.max_number_of_evaluations)


def _get_generated_data(num_images_generated):
  """Get generated images."""
  noise = tf.random.normal([num_images_generated, 64])
  generator_inputs = noise
  generator_fn = networks.generator
  # In order for variables to load, use the same variable scope as in the
  # train job.
  with tf.variable_scope('Generator'):
    data = generator_fn(generator_inputs, is_training=False)

  return data
