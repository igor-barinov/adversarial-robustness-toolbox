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
    """
    This module implements Expectation over Transformation preprocessing for image rotation in PyTorch.
    """

    params = ["nb_samples", "angles", "clip_values", "label_type"]

    label_types = ["classification", "object_detection"]

    def __init__(
        self,
        nb_samples: int,
        clip_values: Tuple[float, float],
        angles: Union[float, Tuple[float, float]] = 45.0,
        label_type: str = 'classification',
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EoTImageRotationPyTorch.

        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
                            for features.
        :param angles: A positive scalar angle in degrees defining the uniform sampling range from negative to
                       positive angles_range.
        :param label_type: String defining the type of labels. Currently supported: `classification`
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super.__init__(apply_fit=apply_fit, apply_predict=apply_predict, clip_values=clip_values, nb_samples=nb_samples)
        
        self.angles = angles
        self.label_type = label_type
        self._check_params()

    def _transform(self, x: "torch.Tensor", y: Optional["torch.Tensor"], **kwargs
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Transformation of an input image and its label by randomly sampled rotation.

        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        import torch
        import torchvision.transforms as T

        if self.label_type == "classification":
            rotater = T.RandomRotation(degrees=self.angles)
            x_preprocess = rotater(x)
            x_preprocess = torch.clamp(input=x_preprocess, min=self.clip_values[0], max=self.clip_values[1])
        elif self.label_type == "object_detection":

        
        return x_preprocess, y

    def _check_params(self) -> None:
        
        # pylint: disable=R0916
        if not isinstance(self.angles, (int, float, tuple)) or (
            isinstance(self.angles, tuple)
            and (
                len(self.angles) != 2
                or not isinstance(self.angles[0], (int, float))
                or not isinstance(self.angles[1], (int, float))
                or self.angles[0] > self.angles[1]
            )
        ):
            raise ValueError("The range of angles must be a float in the range (0.0, 180.0].")

        if self.label_type not in self.label_types:
            raise ValueError(
                "The input for label_type needs to be one of {}, currently receiving `{}`.".format(
                    self.label_types, self.label_type
                )
            )