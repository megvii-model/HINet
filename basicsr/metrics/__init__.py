# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']
