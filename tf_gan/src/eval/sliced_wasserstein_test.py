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
"""Tests for tfgan.eval.sliced_wasserstein."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import ndimage

import tensorflow as tf
import tf_gan as tfgan

from tf_gan.src.eval.sliced_wasserstein import laplacian_pyramid


class ClassifierMetricsTest(tf.test.TestCase):

  def test_laplacian_pyramid(self):
    # The numpy/scipy code for reference estimation comes from:
    # https://github.com/tkarras/progressive_growing_of_gans
    gaussian_filter = np.float32([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                                  [1, 4, 6, 4, 1]]) / 256.0

    def np_pyr_down(minibatch):  # matches cv2.pyrDown()
      assert minibatch.ndim == 4
      return ndimage.convolve(minibatch, gaussian_filter[np.newaxis, np.newaxis, :, :],
                              mode='mirror')[:, :, ::2, ::2]

    def np_pyr_up(minibatch):  # matches cv2.pyrUp()
      assert minibatch.ndim == 4
      s = minibatch.shape
      res = np.zeros((s[0], s[1], s[2] * 2, s[3] * 2), minibatch.dtype)
      res[:, :, ::2, ::2] = minibatch
      return ndimage.convolve(res, gaussian_filter[np.newaxis, np.newaxis, :, :] * 4.0, mode='mirror')

    def np_laplacian_pyramid(minibatch, num_levels):
      # Note: there's a bug in the original SWD, fixed repeatability.
      pyramid = [minibatch.astype('f').copy()]
      for _ in range(1, num_levels):
        pyramid.append(np_pyr_down(pyramid[-1]))
        pyramid[-2] -= np_pyr_up(pyramid[-1])
      return pyramid

    data = np.random.normal(size=[256, 3, 32, 32]).astype('f')
    pyramid = np_laplacian_pyramid(data, 3)
    data_tf = tf.constant(data.transpose(0, 2, 3, 1))
    pyramid_tf = laplacian_pyramid(data_tf, 3)
    with self.cached_session() as sess:
      pyramid_tf = sess.run(pyramid_tf)
    for x in range(3):
      self.assertAllClose(pyramid[x].transpose(0, 2, 3, 1), pyramid_tf[x], atol=1e-6)

  def test_sliced_wasserstein_distance(self):
    """Test the distance."""
    d1 = tf.random.uniform([256, 32, 32, 3])
    d2 = tf.random.normal([256, 32, 32, 3])
    wfunc = tfgan.eval.sliced_wasserstein_distance(d1, d2)
    with self.cached_session() as sess:
      wscores = [sess.run(x) for x in wfunc]
    self.assertAllClose(np.array([0.014, 0.014], 'f'), np.array([x[0] for x in wscores], 'f'), rtol=0.15)
    self.assertAllClose(np.array([0.014, 0.020], 'f'), np.array([x[1] for x in wscores], 'f'), rtol=0.15)

  def test_sliced_wasserstein_distance_svd(self):
    """Test the distance with svd."""
    d1 = tf.random.uniform([256, 32, 32, 3])
    d2 = tf.random.normal([256, 32, 32, 3])
    wfunc = tfgan.eval.sliced_wasserstein_distance(d1, d2, use_svd=True)
    with self.cached_session() as sess:
      wscores = [sess.run(x) for x in wfunc]
    self.assertAllClose(np.array([0.013, 0.013], 'f'), np.array([x[0] for x in wscores], 'f'), rtol=0.15)
    self.assertAllClose(np.array([0.014, 0.019], 'f'), np.array([x[1] for x in wscores], 'f'), rtol=0.15)

  def test_swd_mismatched(self):
    """Test the inputs mismatched shapes are detected."""
    d1 = tf.random.uniform([256, 32, 32, 3])
    d2 = tf.random.normal([256, 32, 31, 3])
    d3 = tf.random.normal([256, 31, 32, 3])
    d4 = tf.random.normal([255, 32, 32, 3])
    with self.assertRaises(ValueError):
      tfgan.eval.sliced_wasserstein_distance(d1, d2)
    with self.assertRaises(ValueError):
      tfgan.eval.sliced_wasserstein_distance(d1, d3)
    with self.assertRaises(ValueError):
      tfgan.eval.sliced_wasserstein_distance(d1, d4)

  def test_swd_not_rgb(self):
    """Test that only RGB is supported."""
    d1 = tf.random.uniform([256, 32, 32, 1])
    d2 = tf.random.normal([256, 32, 32, 1])
    with self.assertRaises(ValueError):
      tfgan.eval.sliced_wasserstein_distance(d1, d2)


if __name__ == '__main__':
  tf.test.main()
