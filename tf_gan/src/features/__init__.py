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
"""TF-GAN features module.

This module includes support for virtual batch normalization, buffer replay,
conditioning, etc.
"""
# pylint: disable=wildcard-import,g-bad-import-order

# Collapse features into a single namespace.
from .clip_weights import *
from .conditioning_utils import *
from .normalization import *
from .random_tensor_pool import *
from .spectral_normalization import *
from .virtual_batchnorm import *

# Collect list of exposed symbols.
from .clip_weights import __all__ as clip_weights_symbols
from .conditioning_utils import __all__ as conditioning_utils_symbols
from .normalization import __all__ as normalization_symbols
from .random_tensor_pool import __all__ as random_tensor_pool_symbols
from .spectral_normalization import __all__ as spectral_normalization_symbols
from .virtual_batchnorm import __all__ as virtual_batchnorm_symbols

__all__ = clip_weights_symbols
__all__ += conditioning_utils_symbols
__all__ += normalization_symbols
__all__ += random_tensor_pool_symbols
__all__ += spectral_normalization_symbols
__all__ += virtual_batchnorm_symbols
