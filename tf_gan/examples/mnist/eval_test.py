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
"""Tests for tfgan.examples.mnist.eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

import eval_lib

mock = tf.test.mock


class EvalTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('RealData', True), ('GeneratedData', False))
  @mock.patch.object(eval_lib.data_provider, 'provide_data', autospec=True)
  @mock.patch.object(eval_lib, 'util', autospec=True)
  def test_build_graph(self, eval_real_images, mock_util, mock_provide_data):
    hparams = eval_lib.HParams(checkpoint_dir='/tmp/mnist/',
                               eval_dir='/tmp/mnist/',
                               dataset_dir=None,
                               num_images_generated=1000,
                               eval_real_images=eval_real_images,
                               noise_dims=64,
                               max_number_of_evaluations=None,
                               write_to_disk=True)

    # Mock input pipeline.
    bs = hparams.num_images_generated
    mock_imgs = np.zeros([bs, 28, 28, 1], dtype=np.float32)
    mock_lbls = np.concatenate((np.ones([bs, 1], dtype=np.int32), np.zeros([bs, 9], dtype=np.int32)), axis=1)
    mock_provide_data.return_value = (mock_imgs, mock_lbls)

    # Mock expensive eval metrics.
    mock_util.mnist_frechet_distance.return_value = 1.0
    mock_util.mnist_score.return_value = 0.0

    eval_lib.evaluate(hparams, run_eval_loop=False)


if __name__ == '__main__':
  tf.test.main()
