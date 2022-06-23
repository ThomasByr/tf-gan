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
"""TF-GAN is a lightweight library for training and evaluating GANs.

In addition to providing the infrastructure for easily training and evaluating
GANS, this library contains modules for a TF-GAN-backed Estimator,
evaluation metrics, features (such as virtual batch normalization), and losses.
Please see README.md for details and usage.

We construct the interface here, and remove undocumented symbols. The high-level
structure should be:
tfgan
-> .estimator
-> .eval
-> .features
-> .losses
  -> .wargs
-> .tpu
"""
# pylint:disable=g-import-not-at-top,g-bad-import-order

# Collapse TF-GAN into a tiered namespace.
# Module names to keep.
from tf_gan.src import estimator
from tf_gan.src import eval  # pylint:disable=redefined-builtin
from tf_gan.src import features
from tf_gan.src import losses
from tf_gan.src import tpu

# Modules to wildcard import.
from tf_gan.src.namedtuples import *  # pylint:disable=wildcard-import
from tf_gan.src.train import *  # pylint:disable=wildcard-import

# Get the version number.
from tf_gan.src.version import __version__

# Collect allowed top-level symbols to expose to users.
__all__ = [
    'estimator',
    'eval',
    'features',
    'losses',
    'tpu',
    '__version__',
]
from tf_gan.src.namedtuples import __all__ as namedtuple_symbols
from tf_gan.src.train import __all__ as train_symbols

__all__ += namedtuple_symbols
__all__ += train_symbols

# Remove undocumented symbols to avoid polluting namespaces.
