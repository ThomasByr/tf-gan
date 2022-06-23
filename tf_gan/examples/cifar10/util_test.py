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
"""Tests for gan.cifar.util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tf_gan.examples.cifar10 import util

mock = tf.test.mock


class UtilTest(tf.test.TestCase):

  def test_get_generator_conditioning(self):
    conditioning = util.get_generator_conditioning(12, 4)
    self.assertEqual([12, 4], conditioning.shape.as_list())

  def test_get_image_grid(self):
    util.get_image_grid(tf.zeros([6, 28, 28, 1]), batch_size=6, num_classes=3, num_images_per_class=1)

  # Mock `inception_score` which is expensive.
  @mock.patch.object(util.tfgan.eval, 'inception_score', autospec=True)
  def test_get_inception_scores(self, mock_inception_score):
    mock_inception_score.return_value = 1.0
    batch_size = 100
    util.get_inception_scores(tf.zeros([batch_size, 28, 28, 3], dtype=tf.float32),
                              batch_size=batch_size,
                              num_inception_images=10)

  # Mock `frechet_inception_distance` which is expensive.
  @mock.patch.object(util.tfgan.eval, 'frechet_inception_distance', autospec=True)
  def test_get_frechet_inception_distance(self, mock_fid):
    mock_fid.return_value = 1.0
    batch_size = 100
    util.get_frechet_inception_distance(tf.zeros([batch_size, 28, 28, 3], dtype=tf.float32),
                                        tf.zeros([batch_size, 28, 28, 3], dtype=tf.float32),
                                        batch_size=batch_size,
                                        num_inception_images=10)


if __name__ == '__main__':
  tf.test.main()
