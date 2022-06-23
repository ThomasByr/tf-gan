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
"""Tests for cifar.train."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
import train_lib

mock = tf.test.mock


class TrainTest(tf.test.TestCase):

  @mock.patch.object(train_lib, 'data_provider', autospec=True)
  def test_build_graph(self, mock_data_provider):
    if tf.executing_eagerly():
      # `tfgan.gan_model` doesn't work when executing eagerly.
      return
    hparams = train_lib.HParams(batch_size=16,
                                max_number_of_steps=0,
                                generator_lr=0.0002,
                                discriminator_lr=0.0002,
                                master='',
                                train_log_dir='/tmp/tfgan_logdir/cifar/',
                                ps_replicas=0,
                                task=0)

    # Mock input pipeline.
    mock_imgs = np.zeros([hparams.batch_size, 32, 32, 3], dtype=np.float32)
    mock_lbls = np.concatenate((np.ones(
        [hparams.batch_size, 1], dtype=np.float32), np.zeros([hparams.batch_size, 9], dtype=np.float32)),
                               axis=1)
    mock_data_provider.provide_data.return_value = (mock_imgs, mock_lbls)

    train_lib.train(hparams)


if __name__ == '__main__':
  tf.test.main()
