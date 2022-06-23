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
"""TF-GAN evaluation module.

This module supports techniques such as Inception Score, Frechet Inception
distance, and Sliced Wasserstein distance.
"""
# pylint: disable=wildcard-import,g-bad-import-order

# Collapse eval into a single namespace.
from .classifier_metrics import *
from .eval_utils import *
from .inception_metrics import *
from .sliced_wasserstein import *
from .summaries import *

# Collect list of exposed symbols.
from .classifier_metrics import __all__ as classifier_metrics_symbols
from .eval_utils import __all__ as eval_utils_symbols
from .inception_metrics import __all__ as inception_metrics_symbols
from .sliced_wasserstein import __all__ as sliced_wasserstein_symbols
from .summaries import __all__ as summaries_symbols

__all__ = classifier_metrics_symbols
__all__ += eval_utils_symbols
__all__ += inception_metrics_symbols
__all__ += sliced_wasserstein_symbols
__all__ += summaries_symbols
