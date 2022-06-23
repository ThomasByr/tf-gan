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
"""Evaluates a conditional TF-GAN trained MNIST model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow.compat.v1 as tf
import conditional_eval_lib

flags.DEFINE_string('checkpoint_dir', '/tmp/mnist/', 'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '/tmp/mnist/', 'Directory where the results are saved to.')

flags.DEFINE_integer('num_images_per_class', 10, 'Number of images to generate per class.')

flags.DEFINE_integer('noise_dims', 64, 'Dimensions of the generator noise vector')

flags.DEFINE_integer('max_number_of_evaluations', None, 'Number of times to run evaluation. If `None`, run '
                     'forever.')

flags.DEFINE_boolean('write_to_disk', True, 'If `True`, run images to disk.')

FLAGS = flags.FLAGS


def main(_):
  hparams = conditional_eval_lib.HParams(FLAGS.checkpoint_dir, FLAGS.eval_dir, FLAGS.num_images_per_class,
                                         FLAGS.noise_dims, FLAGS.max_number_of_evaluations,
                                         FLAGS.write_to_disk)
  conditional_eval_lib.evaluate(hparams, run_eval_loop=True)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  app.run(main)
