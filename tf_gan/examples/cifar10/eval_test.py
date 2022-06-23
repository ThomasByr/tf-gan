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
"""Tests for CIFAR10 eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow.compat.v1 as tf
from tf_gan.examples.cifar10 import eval_lib

mock = tf.test.mock


class EvalTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {'eval_real_images': True},
      {
          'eval_real_images': False,
      },
  )
  @mock.patch.object(eval_lib.util, 'get_frechet_inception_distance', autospec=True)
  @mock.patch.object(eval_lib.util, 'get_inception_scores', autospec=True)
  @mock.patch.object(eval_lib.data_provider, 'provide_data', autospec=True)
  def test_build_graph(self, mock_provide_data, mock_iscore, mock_fid, eval_real_images):
    hparams = eval_lib.HParams(master='',
                               checkpoint_dir='/tmp/cifar10/',
                               eval_dir='/tmp/cifar10/',
                               num_images_generated=100,
                               num_inception_images=10,
                               eval_real_images=eval_real_images,
                               eval_frechet_inception_distance=True,
                               max_number_of_evaluations=None,
                               write_to_disk=True)

    # Mock reads from disk.
    mock_provide_data.return_value = (tf.ones([hparams.num_images_generated, 32, 32,
                                               3]), tf.zeros([hparams.num_images_generated]))

    # Mock `frechet_inception_distance` and `inception_score`, which are
    # expensive.
    mock_fid.return_value = 1.0
    mock_iscore.return_value = 1.0

    eval_lib.evaluate(hparams, run_eval_loop=False)


if __name__ == '__main__':
  tf.test.main()
