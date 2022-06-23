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
"""Neural network for a StarGAN model.

This module contains the Generator and Discriminator Neural Network to build a
StarGAN model.

See https://arxiv.org/abs/1711.09020 for details about the model.

See https://github.com/yunjey/StarGAN for the original pytorch implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

import layers
import ops


def generator(inputs, targets):
  """Generator module.

  Piece everything together for the Generator.

  PyTorch Version:
  https://github.com/yunjey/StarGAN/blob/fbdb6a6ce2a4a92e1dc034faec765e0dbe4b8164/model.py#L22

  Args:
    inputs: Tensor of shape (batch_size, h, w, c) representing the
      images/information that we want to transform.
    targets: Tensor of shape (batch_size, num_domains) representing the target
      domain the generator should transform the image/information to.

  Returns:
    Tensor of shape (batch_size, h, w, c) as the inputs.
  """

  with tf.variable_scope('generator'):

    input_with_condition = ops.condition_input_with_pixel_padding(inputs, targets)

    down_sample = layers.generator_down_sample(input_with_condition)

    bottleneck = layers.generator_bottleneck(down_sample)

    up_sample = layers.generator_up_sample(bottleneck, inputs.shape[-1])

  return up_sample


def discriminator(input_net, class_num):
  """Discriminator Module.

  Piece everything together and reshape the output source tensor

  PyTorch Version:
  https://github.com/yunjey/StarGAN/blob/fbdb6a6ce2a4a92e1dc034faec765e0dbe4b8164/model.py#L63

  Notes:
  The PyTorch Version run the reduce_mean operation later in their solver:
  https://github.com/yunjey/StarGAN/blob/fbdb6a6ce2a4a92e1dc034faec765e0dbe4b8164/solver.py#L245

  Args:
    input_net: Tensor of shape (batch_size, h, w, c) as batch of images.
    class_num: (int) number of domain to be predicted

  Returns:
    output_src: Tensor of shape (batch_size) where each value is a logit
    representing whether the image is real of fake.
    output_cls: Tensor of shape (batch_size, class_um) where each value is a
    logit representing whether the image is in the associated domain.
  """

  with tf.variable_scope('discriminator'):

    hidden = layers.discriminator_input_hidden(input_net)

    output_src = layers.discriminator_output_source(hidden)
    output_src = tf.layers.flatten(output_src)
    output_src = tf.reduce_mean(input_tensor=output_src, axis=1)

    output_cls = layers.discriminator_output_class(hidden, class_num)

  return output_src, output_cls
