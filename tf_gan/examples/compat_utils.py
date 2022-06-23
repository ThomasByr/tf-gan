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
"""Utilities for running tests in a TF 1.x and 2.x compatible way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def crop_and_resize(*args, **kwargs):
  """`tf.image.crop_and_resize` that works for TF 1.x and 2.x."""
  try:
    return tf.image.crop_and_resize(*args, **kwargs)
  except (TypeError, AttributeError):
    if 'box_ind' in kwargs:
      kwargs['box_indices'] = kwargs['box_ind']
      del kwargs['box_ind']
    return tf.image.crop_and_resize(*args, **kwargs)


def nn_avg_pool2d(*args, **kwargs):
  """`tf.nn.avg_pool2d` that works for TF 1.x and 2.x."""
  try:
    return tf.nn.avg_pool2d(*args, **kwargs)
  except (TypeError, AttributeError):
    if 'input' in kwargs:
      kwargs['value'] = kwargs['input']
      del kwargs['input']
    return tf.nn.avg_pool(*args, **kwargs)


def batch_to_space(*args, **kwargs):
  """`tf.batch_to_space` that works for TF 1.x and 2.x."""
  try:
    return tf.batch_to_space(*args, **kwargs)
  except TypeError:
    if 'block_shape' in kwargs:
      kwargs['block_size'] = kwargs['block_shape']
      del kwargs['block_shape']
    return tf.batch_to_space(*args, **kwargs)
