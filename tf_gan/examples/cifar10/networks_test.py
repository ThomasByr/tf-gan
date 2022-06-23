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
"""Tests for CIFAR10 networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from tf_gan.examples.cifar10 import networks


class NetworksTest(tf.test.TestCase):

  def test_generator(self):
    tf.set_random_seed(1234)
    batch_size = 100
    noise = tf.random.normal([batch_size, 64])
    image = networks.generator(noise)
    with self.cached_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      image_np = sess.run(image)

    self.assertAllEqual([batch_size, 32, 32, 3], image_np.shape)
    self.assertTrue(np.all(np.abs(image_np) <= 1))

  def test_discriminator(self):
    batch_size = 5
    image = tf.random.uniform([batch_size, 32, 32, 3], -1, 1)
    dis_output = networks.discriminator(image, None)
    with self.cached_session(use_gpu=True) as sess:
      sess.run(tf.global_variables_initializer())
      dis_output_np = sess.run(dis_output)

    self.assertAllEqual([batch_size, 1], dis_output_np.shape)


if __name__ == '__main__':
  tf.test.main()
