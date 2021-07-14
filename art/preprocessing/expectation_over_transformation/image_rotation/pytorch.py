# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements Expectation over Transformation preprocessing for image rotation in PyTorch.
"""

from typing import Optional, Tuple

from art.preprocessing.expectation_over_transformation.pytorch import EoTPyTorch

class EoTImageRotationPyTorch(EoTPyTorch):
    def __init__(
        self,
        nb_samples: int,
        clip_values: Tuple[float, float],
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        super.__init__(apply_fit=apply_fit, apply_predict=apply_predict, clip_values=clip_values, nb_samples=nb_samples)
        
        self._check_params()

    def _transform(self, x: "torch.Tensor", y: Optional["torch.Tensor"], **kwargs
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        pass

    def _check_params(self) -> None:
        pass