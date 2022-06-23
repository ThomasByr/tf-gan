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
"""Tests for stargan.data_provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
import tensorflow.compat.v1 as tf

import data_provider

mock = tf.test.mock


class DataProviderTest(tf.test.TestCase, absltest.TestCase):

  def setUp(self):
    super(DataProviderTest, self).setUp()
    mock_imgs = np.zeros([128, 128, 3], dtype=np.uint8)
    self.mock_ds = tf.data.Dataset.from_tensors({
        'attributes': {
            'A': True,
            'B': True,
            'C': True
        },
        'image': mock_imgs
    })

  @mock.patch.object(data_provider, 'tfds', autospec=True)
  def test_provide_data(self, mock_tfds):
    batch_size = 5
    patch_size = 32
    mock_tfds.load.return_value = self.mock_ds

    images, labels = data_provider.provide_data('test',
                                                batch_size,
                                                patch_size=patch_size,
                                                domains=('A', 'B', 'C'))
    self.assertLen(images, 3)
    self.assertLen(labels, 3)

    with self.cached_session() as sess:
      images = sess.run(images)
      labels = sess.run(labels)
    for img in images:
      self.assertTupleEqual(img.shape, (batch_size, patch_size, patch_size, 3))
      self.assertTrue(np.all(np.abs(img) <= 1))
    for lbl in labels:
      expected_lbls_shape = (batch_size, 3)
      self.assertTupleEqual(lbl.shape, expected_lbls_shape)


if __name__ == '__main__':
  tf.test.main()
