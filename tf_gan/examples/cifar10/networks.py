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
"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def _leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=0.2)


def _batch_norm(x, is_training, name):
  return tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-5, training=is_training, name=name)


def _dense(x, channels, name):
  return tf.layers.dense(x,
                         channels,
                         kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                         name=name)


def _conv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d(x,
                          filters, [kernel_size, kernel_size],
                          strides=[stride, stride],
                          padding='same',
                          kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                          name=name)


def _deconv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d_transpose(x,
                                    filters, [kernel_size, kernel_size],
                                    strides=[stride, stride],
                                    padding='same',
                                    kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                                    name=name)


def discriminator(images, unused_conditioning, is_training=True, scope='Discriminator'):
  """Discriminator for CIFAR images.

  Args:
    images: A Tensor of shape [batch size, width, height, channels], that can be
      either real or generated. It is the discriminator's goal to distinguish
      between the two.
    unused_conditioning: The TFGAN API can help with conditional GANs, which
      would require extra `condition` information to both the generator and the
      discriminator. Since this example is not conditional, we do not use this
      argument.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.
    scope: A variable scope or string for the discriminator.

  Returns:
    A 1D Tensor of shape [batch size] representing the confidence that the
    images are real. The output can lie in [-inf, inf], with positive values
    indicating high confidence that the images are real.
  """
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x = _conv2d(images, 64, 5, 2, name='d_conv1')
    x = _leaky_relu(x)

    x = _conv2d(x, 128, 5, 2, name='d_conv2')
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn2'))

    x = _conv2d(x, 256, 5, 2, name='d_conv3')
    x = _leaky_relu(_batch_norm(x, is_training, name='d_bn3'))

    x = tf.reshape(x, [-1, 4 * 4 * 256])

    x = _dense(x, 1, name='d_fc_4')

    return x


def generator(noise, is_training=True, scope='Generator'):
  """Generator to produce CIFAR images.

  Args:
    noise: A 2D Tensor of shape [batch size, noise dim]. Since this example
      does not use conditioning, this Tensor represents a noise vector of some
      kind that will be reshaped by the generator into CIFAR examples.
    is_training: If `True`, batch norm uses batch statistics. If `False`, batch
      norm uses the exponential moving average collected from population
      statistics.
    scope: A variable scope or string for the generator.

  Returns:
    A single Tensor with a batch of generated CIFAR images.
  """
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    net = _dense(noise, 4096, name='g_fc1')
    net = tf.nn.relu(_batch_norm(net, is_training, name='g_bn1'))

    net = tf.reshape(net, [-1, 4, 4, 256])

    net = _deconv2d(net, 128, 5, 2, name='g_dconv2')
    net = tf.nn.relu(_batch_norm(net, is_training, name='g_bn2'))

    net = _deconv2d(net, 64, 4, 2, name='g_dconv3')
    net = tf.nn.relu(_batch_norm(net, is_training, name='g_bn3'))

    net = _deconv2d(net, 3, 4, 2, name='g_dconv4')
    net = tf.tanh(net)

    return net
