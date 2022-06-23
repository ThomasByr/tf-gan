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
"""Contains code for loading and preprocessing the CIFAR data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def provide_dataset(split, batch_size, num_parallel_calls=None, shuffle=True, one_hot=True):
  """Provides batches of CIFAR10 digits.

  Args:
    split: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.
    one_hot: Output one hot vector instead of int32 label.

  Returns:
    A tf.data.Dataset with:
      * images: A `Tensor` of size [batch_size, 32, 32, 3] and type tf.float32.
          Output pixel values are in [-1, 1].
      * labels: A `Tensor` of size [batch_size, 10] of one-hot label
          encodings with type tf.int32, or a `Tensor` of size [batch_size],
          depending on the value of `one_hot`.

  Raises:
    ValueError: If `split` isn't `train` or `test`.
  """
  ds = tfds.load('cifar10:3.*.*', split=split, shuffle_files=shuffle)

  def _preprocess(element):
    """Map elements to the example dicts expected by the model."""
    images = (tf.cast(element['image'], tf.float32) - 128.0) / 128.0
    labels = element['label']
    if one_hot:
      num_classes = 10
      labels = tf.one_hot(labels, num_classes)
    else:
      labels = tf.cast(labels, tf.float32)
    return {'images': images, 'labels': labels}

  ds = (ds.map(_preprocess, num_parallel_calls=num_parallel_calls).cache().repeat())
  if shuffle:
    ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
  ds = (ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

  return ds


def provide_data(split, batch_size, num_parallel_calls=None, shuffle=True, one_hot=True):
  """Provides batches of CIFAR10 digits.

  Args:
    split: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.
    one_hot: Output one hot vector instead of int32 label.

  Returns:
    images: A `Tensor` of size [batch_size, 32, 32, 3]. Output pixel values are
      in [-1, 1].
    labels: A `Tensor` of size [batch_size, 10] of one-hot label
      encodings with type tf.int32, or a `Tensor` of size [batch_size],
      depending on the value of `one_hot`.

  Raises:
    ValueError: If `split` isn't `train` or `test`.
  """
  ds = provide_dataset(split, batch_size, num_parallel_calls, shuffle, one_hot)

  next_batch = tf.data.make_one_shot_iterator(ds).get_next()
  images, labels = next_batch['images'], next_batch['labels']

  return images, labels


def float_image_to_uint8(image):
  """Convert float image in [-1, 1) to [0, 255] uint8.

  Note that `1` gets mapped to `0`, but `1 - epsilon` gets mapped to 255.

  Args:
    image: An image tensor. Values should be in [-1, 1).

  Returns:
    Input image cast to uint8 and with integer values in [0, 255].
  """
  image = (image*128.0) + 128.0
  return tf.cast(image, tf.uint8)
