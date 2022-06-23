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
"""TF-GAN TPU support."""
# pylint: disable=wildcard-import,g-bad-import-order,line-too-long,g-no-space-after-comment

# Collapse losses into a single namespace.
# TODO(joelshor): Figure out why including this line breaks the open source
# build
# from tensorflow_gan.python.tpu.cross_replica_ops import *
from tf_gan.src.tpu.normalization_ops import *

# Collect list of exposed symbols.
#from tensorflow_gan.python.tpu.cross_replica_ops import __all__ as cross_replica_ops_symbols
from tf_gan.src.tpu.normalization_ops import __all__ as normalization_ops_symbols

__all__ = normalization_ops_symbols
# __all__ += cross_replica_ops_symbols
