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

import network


class NetworkTest(tf.test.TestCase):

  def test_generator(self):

    n = 2
    h = 128
    w = h
    c = 4
    class_num = 3

    input_tensor = tf.random.uniform((n, h, w, c))
    target_tensor = tf.random.uniform((n, class_num))
    output_tensor = network.generator(input_tensor, target_tensor)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(output_tensor)
      self.assertTupleEqual((n, h, w, c), output.shape)

  def test_discriminator(self):

    n = 2
    h = 128
    w = h
    c = 3
    class_num = 3

    input_tensor = tf.random.uniform((n, h, w, c))
    output_src_tensor, output_cls_tensor = network.discriminator(input_tensor, class_num)

    with self.cached_session() as sess:
      sess.run(tf.global_variables_initializer())
      output_src, output_cls = sess.run([output_src_tensor, output_cls_tensor])
      self.assertEqual(1, len(output_src.shape))
      self.assertEqual(n, output_src.shape[0])
      self.assertTupleEqual((n, class_num), output_cls.shape)


if __name__ == '__main__':
  tf.test.main()
