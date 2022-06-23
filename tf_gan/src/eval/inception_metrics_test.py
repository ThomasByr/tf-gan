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
"""Tests for TF-GAN internal inception_metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized
import tensorflow as tf
import tf_gan as tfgan

mock = tf.compat.v1.test.mock


class FakeInceptionModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  def __call__(self, x):
    bs = tf.shape(x)[0]
    logits = tf.zeros([bs, 1008])
    pool_3 = tf.ones([bs, 2048])
    return {'logits': logits, 'pool_3': pool_3}


class RunInceptionTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RunInceptionTest, self).setUp()
    self.export_path = os.path.join(self.get_temp_dir(), 'my-module')
    tf.saved_model.save(FakeInceptionModule(), self.export_path)
    classifier_fn = tfgan.eval.classifier_fn_from_tfhub(self.export_path, None)

    def run_inception(*args, **kwargs):
      return tfgan.eval.run_inception(*args, classifier_fn=classifier_fn, **kwargs)

    self.run_inception = run_inception

  @parameterized.parameters(
      {'num_batches': 1},
      {'num_batches': 4},
  )
  def test_run_inception_graph(self, num_batches):
    """Test `run_inception` graph construction."""
    batch_size = 8
    img = tf.ones([batch_size, 299, 299, 3])

    results = self.run_inception(img, num_batches=num_batches)

    self.assertIsInstance(results, dict)
    self.assertLen(results, 2)

    self.assertIn('logits', results)
    logits = results['logits']
    self.assertIsInstance(logits, tf.Tensor)
    logits.shape.assert_is_compatible_with([batch_size, 1008])

    self.assertIn('pool_3', results)
    pool = results['pool_3']
    self.assertIsInstance(pool, tf.Tensor)
    pool.shape.assert_is_compatible_with([batch_size, 2048])

    # Check that none of the model variables are trainable.
    self.assertListEqual([], tf.compat.v1.trainable_variables())

  def test_run_inception_multicall(self):
    """Test that `run_inception` can be called multiple times."""
    for batch_size in (7, 3, 2):
      img = tf.ones([batch_size, 299, 299, 3])
      self.run_inception(img)


class SampleAndRunInception(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(SampleAndRunInception, self).setUp()
    self.export_path = os.path.join(self.get_temp_dir(), 'my-module')
    tf.saved_model.save(FakeInceptionModule(), self.export_path)
    classifier_fn = tfgan.eval.classifier_fn_from_tfhub(self.export_path, None)

    def sample_and_run_inception(*args, **kwargs):
      return tfgan.eval.sample_and_run_inception(*args, classifier_fn=classifier_fn, **kwargs)

    self.sample_and_run_inception = sample_and_run_inception

  @parameterized.parameters(
      {'num_batches': 1},
      {'num_batches': 4},
  )
  def test_sample_and_run_inception_graph(self, num_batches):
    """Test `sample_and_run_inception` graph construction."""
    batch_size = 8

    def sample_fn(_):
      return tf.ones([batch_size, 244, 244, 3])

    sample_inputs = [1] * num_batches

    results = self.sample_and_run_inception(sample_fn, sample_inputs)

    self.assertIsInstance(results, dict)
    self.assertLen(results, 2)

    self.assertIn('logits', results)
    logits = results['logits']
    self.assertIsInstance(logits, tf.Tensor)
    logits.shape.assert_is_compatible_with([batch_size * num_batches, 1008])

    self.assertIn('pool_3', results)
    pool = results['pool_3']
    self.assertIsInstance(pool, tf.Tensor)
    pool.shape.assert_is_compatible_with([batch_size * num_batches, 2048])

    # Check that none of the model variables are trainable.
    self.assertListEqual([], tf.compat.v1.trainable_variables())


class InceptionScore(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(InceptionScore, self).setUp()
    self.export_path = os.path.join(self.get_temp_dir(), 'my-module')
    tf.saved_model.save(FakeInceptionModule(), self.export_path)
    classifier_fn = tfgan.eval.classifier_fn_from_tfhub(self.export_path, tfgan.eval.INCEPTION_OUTPUT, True)

    def inception_score(*args, **kwargs):
      return tfgan.eval.inception_score(*args, classifier_fn=classifier_fn, **kwargs)

    self.inception_score = inception_score

    def inception_score_streaming(*args, **kwargs):
      return tfgan.eval.inception_score_streaming(*args, classifier_fn=classifier_fn, **kwargs)

    self.inception_score_streaming = inception_score_streaming

  @parameterized.parameters(
      {
          'num_batches': 1,
          'streaming': True
      },
      {
          'num_batches': 1,
          'streaming': False
      },
      {
          'num_batches': 3,
          'streaming': True
      },
      {
          'num_batches': 3,
          'streaming': False
      },
  )
  def test_inception_score_graph(self, num_batches, streaming):
    """Test `inception_score` graph construction."""
    if streaming and tf.executing_eagerly():
      # streaming doesn't work in eager execution.
      return
    img = tf.zeros([6, 299, 299, 3])
    if streaming:
      score, update_op = self.inception_score_streaming(img, num_batches=num_batches)
      self.assertIsInstance(update_op, tf.Tensor)
      update_op.shape.assert_has_rank(0)
    else:
      score = self.inception_score(img, num_batches=num_batches)

    self.assertIsInstance(score, tf.Tensor)
    score.shape.assert_has_rank(0)

    # Check that none of the model variables are trainable.
    self.assertEmpty(tf.compat.v1.trainable_variables())


class FrechetInceptionDistance(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(FrechetInceptionDistance, self).setUp()
    self.export_path = os.path.join(self.get_temp_dir(), 'my-module')
    tf.saved_model.save(FakeInceptionModule(), self.export_path)
    classifier_fn = tfgan.eval.classifier_fn_from_tfhub(self.export_path, tfgan.eval.INCEPTION_FINAL_POOL,
                                                        True)

    def frechet_inception_distance(*args, **kwargs):
      return tfgan.eval.frechet_inception_distance(*args, classifier_fn=classifier_fn, **kwargs)

    self.frechet_inception_distance = frechet_inception_distance

    def fid_streaming(*args, **kwargs):
      return tfgan.eval.frechet_inception_distance_streaming(*args, classifier_fn=classifier_fn, **kwargs)

    self.frechet_inception_distance_streaming = fid_streaming

  @parameterized.parameters(
      {
          'num_batches': 1,
          'streaming': True
      },
      {
          'num_batches': 1,
          'streaming': False
      },
      {
          'num_batches': 3,
          'streaming': True
      },
      {
          'num_batches': 3,
          'streaming': False
      },
  )
  def test_frechet_inception_distance_graph(self, num_batches, streaming):
    """Test `frechet_inception_distance` graph construction."""
    if streaming and tf.executing_eagerly():
      # streaming doesn't work in eager execution.
      return
    img = tf.ones([6, 299, 299, 3])

    if streaming:
      distance, update_op = self.frechet_inception_distance_streaming(img, img, num_batches=num_batches)
      self.assertIsInstance(update_op, tf.Tensor)
      update_op.shape.assert_has_rank(0)
    else:
      distance = self.frechet_inception_distance(img, img, num_batches=num_batches)

    self.assertIsInstance(distance, tf.Tensor)
    distance.shape.assert_has_rank(0)

    # Check that none of the model variables are trainable.
    self.assertEmpty(tf.compat.v1.trainable_variables())


if __name__ == '__main__':
  tf.test.main()
