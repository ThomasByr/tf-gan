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

import tensorflow.compat.v1 as tf

import layers


class LayersTest(tf.test.TestCase):

  def test_residual_block(self):

    n = 2
    h = 32
    w = h
    c = 256

    input_tensor = tf.random.uniform((n, h, w, c))
    output_tensor = layers._residual_block(input_net=input_tensor,
                                           num_outputs=c,
                                           kernel_size=3,
                                           stride=1,
                                           padding_size=1)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h, w, c), output.shape)

  def test_generator_down_sample(self):

    n = 2
    h = 128
    w = h
    c = 3 + 3

    input_tensor = tf.random.uniform((n, h, w, c))
    output_tensor = layers.generator_down_sample(input_tensor)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h // 4, w // 4, 256), output.shape)

  def test_generator_bottleneck(self):

    n = 2
    h = 32
    w = h
    c = 256

    input_tensor = tf.random.uniform((n, h, w, c))
    output_tensor = layers.generator_bottleneck(input_tensor)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h, w, c), output.shape)

  def test_generator_up_sample(self):

    n = 2
    h = 32
    w = h
    c = 256
    c_out = 3

    input_tensor = tf.random.uniform((n, h, w, c))
    output_tensor = layers.generator_up_sample(input_tensor, c_out)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h * 4, w * 4, c_out), output.shape)

  def test_discriminator_input_hidden(self):

    n = 2
    h = 128
    w = 128
    c = 3

    input_tensor = tf.random.uniform((n, h, w, c))
    output_tensor = layers.discriminator_input_hidden(input_tensor)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, 2, 2, 2048), output.shape)

  def test_discriminator_output_source(self):

    n = 2
    h = 2
    w = 2
    c = 2048

    input_tensor = tf.random.uniform((n, h, w, c))
    output_tensor = layers.discriminator_output_source(input_tensor)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h, w, 1), output.shape)

  def test_discriminator_output_class(self):

    n = 2
    h = 2
    w = 2
    c = 2048
    num_domain = 3

    input_tensor = tf.random.uniform((n, h, w, c))
    output_tensor = layers.discriminator_output_class(input_tensor, num_domain)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, num_domain), output.shape)


if __name__ == '__main__':
  tf.test.main()
