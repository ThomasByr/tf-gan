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
r"""Model evaluation tools for TF-GAN.

These methods come from https://arxiv.org/abs/1606.03498 and
https://arxiv.org/abs/1706.08500.

NOTE: This implementation uses the same weights as in
https://github.com/openai/improved-gan/blob/master/inception_score/model.py,
.


Note that the default checkpoint is the same as in the OpenAI implementation
(https://github.com/openai/improved-gan/tree/master/inception_score), but is
more numerically stable and is an unbiased estimator of the true Inception score
even when splitting the inputs into batches. Also, the graph modified so that it
works with arbitrary batch size and the preprocessing moved to the `preprocess`
function. Note that the modifications in the GitHub implementation are *not*
sufficient to run with arbitrary batch size, due to the hardcoded resize value.

The graph runs on TPU.

Finally, I manually removed the placeholder input, which was unnecessary and is
not supported on TPU.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

import tensorflow as tf
from tf_gan.src.eval import classifier_metrics
import tensorflow_hub as tfhub

__all__ = [
    'classifier_fn_from_tfhub',
    'run_inception',
    'sample_and_run_inception',
    'inception_score',
    'inception_score_streaming',
    'frechet_inception_distance',
    'frechet_inception_distance_streaming',
    'kernel_inception_distance',
    'kernel_inception_distance_and_std',
    'INCEPTION_TFHUB',
    'INCEPTION_OUTPUT',
    'INCEPTION_FINAL_POOL',
    'INCEPTION_DEFAULT_IMAGE_SIZE',
]

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {INCEPTION_OUTPUT: tf.float32, INCEPTION_FINAL_POOL: tf.float32}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def classifier_fn_from_tfhub(tfhub_module, output_fields, return_tensor=False):
  """Returns a function that can be as a classifier function.

  Wrapping the TF-Hub module in another function defers loading the module until
  use, which is useful for mocking and not computing heavy default arguments.

  Args:
    tfhub_module: A string handle for a TF-Hub module.
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]

  def _classifier_fn(images):
    output = tfhub.load(tfhub_module)(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

  return _classifier_fn


run_inception = functools.partial(classifier_metrics.run_classifier_fn,
                                  classifier_fn=classifier_fn_from_tfhub(INCEPTION_TFHUB, None),
                                  dtypes=_DEFAULT_DTYPES)

sample_and_run_inception = functools.partial(classifier_metrics.sample_and_run_classifier_fn,
                                             classifier_fn=classifier_fn_from_tfhub(INCEPTION_TFHUB, None),
                                             dtypes=_DEFAULT_DTYPES)

inception_score = functools.partial(classifier_metrics.classifier_score,
                                    classifier_fn=classifier_fn_from_tfhub(INCEPTION_TFHUB, INCEPTION_OUTPUT,
                                                                           True))

inception_score_streaming = functools.partial(classifier_metrics.classifier_score_streaming,
                                              classifier_fn=classifier_fn_from_tfhub(
                                                  INCEPTION_TFHUB, INCEPTION_OUTPUT, True))

frechet_inception_distance = functools.partial(classifier_metrics.frechet_classifier_distance,
                                               classifier_fn=classifier_fn_from_tfhub(
                                                   INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True))

frechet_inception_distance_streaming = functools.partial(
    classifier_metrics.frechet_classifier_distance_streaming,
    classifier_fn=classifier_fn_from_tfhub(INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True))

kernel_inception_distance = functools.partial(classifier_metrics.kernel_classifier_distance,
                                              classifier_fn=classifier_fn_from_tfhub(
                                                  INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True))

kernel_inception_distance_and_std = functools.partial(classifier_metrics.kernel_classifier_distance_and_std,
                                                      classifier_fn=classifier_fn_from_tfhub(
                                                          INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True))
