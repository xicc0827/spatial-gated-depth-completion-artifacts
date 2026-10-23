"""Explicit torchvision port of CompletionFormer's DCN call interface.

This is NOT the original CUDA extension. Validate against test_completionformer_dcn
before use; full pretrained-model parity remains a separate requirement.
"""
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class ModulatedDeformConvFunction:
    @staticmethod
    def apply(input, offset, mask, weight, bias, stride, padding, dilation,
              groups, deformable_groups, im2col_step):
        kh, kw = weight.shape[-2:]
        if input.shape[1] != weight.shape[1] * groups:
            raise ValueError("Convolution group mismatch")
        if offset.shape[1] != 2 * kh * kw * deformable_groups:
            raise ValueError("Offset group mismatch")
        if mask.shape[1] != kh * kw * deformable_groups:
            raise ValueError("Mask group mismatch")
        # torchvision infers both group counts from shapes. im2col_step is an
        # execution chunk-size in the original extension, not a model parameter.
        return deform_conv2d(input, offset, weight, bias=bias,
                             stride=_pair(stride), padding=_pair(padding),
                             dilation=_pair(dilation), mask=mask)
