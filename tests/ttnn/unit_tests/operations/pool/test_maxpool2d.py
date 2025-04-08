# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import itertools
import random
import torch
import pytest
import math
from typing import Optional, Tuple, List

from models.utility_functions import is_wormhole_b0, is_grayskull, is_x2_harvested, torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.max_pool2d_common import run_max_pool2d, mesh_device_fixture

import ttnn

parameters = {
    "max_pool2d_short_sweep_suite": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [1, 128, 112, 112, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 128, 150, 150, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 128, 56, 56, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 128, 64, 64, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 16, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 192, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 192, 56, 56, 3, 3, 2, 2, 0, 0, 1, 1, True],
            [1, 256, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 256, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 56, 56, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 256, 75, 75, 2, 2, 2, 2, 0, 0, 1, 1, True],
            [1, 32, 256, 256, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 320, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 4, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],  # requires padding along C
            [1, 480, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 480, 28, 28, 3, 3, 2, 2, 0, 0, 1, 1, True],
            [1, 512, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 512, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 512, 19, 19, 3, 3, 1, 1, 1, 1, 1, 1, False],
            [1, 512, 28, 28, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 512, 38, 38, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 528, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, True],  # required padding along C
            [1, 64, 112, 112, 3, 3, 2, 2, 0, 0, 1, 1, True],
            [1, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 64, 128, 128, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 64, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 64, 24, 24, 2, 2, 1, 1, 0, 0, 1, 1, False],
            [1, 64, 300, 300, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 64, 360, 640, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 64, 400, 544, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 640, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 832, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, True],
            [1, 832, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, True],
            [1, 96, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
    "test_run_max_pool": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [1, 32, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],  # functional_unet
            # [1, 64, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 3, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False],  # vgg
            # [1, 512, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            # [1, 512, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # [1, 512, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            [1, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],  # resnet
            [2, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [4, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [8, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [1, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
    "test_run_max_pool_width_shard": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            # [1, 32, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False], # functional_unet
            # [1, 64, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],
            # [1, 3, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False], # vgg
            [1, 32768, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            [1, 32768, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # [1, 32768, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            [1, 6144, 6, 6, 5, 5, 1, 1, 2, 2, 1, 1, False],
            [1, 6144, 6, 6, 9, 9, 1, 1, 2, 2, 1, 1, False],
            [1, 6144, 6, 6, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # [1, 6144, 6, 6, 13, 13, 1, 1, 2, 2, 1, 1, False],
            # [1, 6144, 6, 6, 13, 13, 1, 1, 4, 4, 1, 1, False],
            # [1, 6144, 6, 6, 13, 13, 1, 1, 6, 6, 1, 1, False],
            # [1, 512, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            # [1, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False], #resnet
            # [2, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            # [4, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            # [8, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
    "test_run_max_pool_height_shard": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            # [1, 32768, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            # [1, 32768, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # [1, 32768, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            # [1, 6144, 6, 6, 5, 5, 1, 1, 2, 2, 1, 1, False],
            # [1, 6144, 6, 6, 9, 9, 1, 1, 2, 2, 1, 1, False],
            # [1, 6144, 6, 6, 9, 9, 1, 1, 4, 4, 1, 1, False],
            [1, 512, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            [1, 512, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            [1, 512, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            [1, 256, 20, 20, 8, 8, 6, 6, 0, 0, 1, 1, False],
        ],
    },
    "test_run_max_pool_block_shard": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "in_place": [True, False],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            # [1, 32, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],  # functional_unet
            # [1, 64, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],
            # [1, 3, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False],  # vgg
            [1, 512, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            [1, 512, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            [1, 512, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            # [1, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],  # resnet
            # [2, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            # [4, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            # [8, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
    "test_run_max_pool_mem_config": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [batch_size, input_channels, input_height, input_width, kernel_height, kernel_width, stride_h, strid_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            # [1, 32, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],  # functional_unet
            # [1, 64, 1056, 160, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 3, 224, 224, 2, 2, 2, 2, 0, 0, 1, 1, False],  # vgg
            # [1, 512, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, False],  # yolo
            # [1, 512, 10, 10, 9, 9, 1, 1, 4, 4, 1, 1, False],
            # [1, 512, 10, 10, 13, 13, 1, 1, 6, 6, 1, 1, False],
            [1, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],  # resnet
            [2, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [4, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
            [8, 3, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    input_specs,
    dtype,
    *,
    device,
):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_specs
    sharding = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    return run_max_pool2d(
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding,
        ceil_mode,
    )


import pytest


@pytest.mark.parametrize("input_spec", parameters["max_pool2d_short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["max_pool2d_short_sweep_suite"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_localrun(device, dtype, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_height_shard"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_height_shard"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["test_run_max_pool_height_shard"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_localrun(device, dtype, in_place, input_spec):
    if dtype == ttnn.bfloat8_b and in_place:
        pytest.xfail("BFloat8 is not currently supported when using in-place halo")
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool(device, dtype, input_spec):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_width_shard"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_width_shard"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["test_run_max_pool_width_shard"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool_width_shard(device, dtype, in_place, input_spec):
    if dtype == ttnn.bfloat8_b and in_place:
        pytest.xfail("BFloat8 is not currently supported when using in-place halo")
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_block_shard"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_block_shard"]["dtype"])
@pytest.mark.parametrize("in_place", parameters["test_run_max_pool_block_shard"]["in_place"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool_block_shard(device, dtype, in_place, input_spec):
    if dtype == ttnn.bfloat8_b and in_place:
        pytest.xfail("BFloat8 is not currently supported when using in-place halo")
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        sharding=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ceil_mode=ceil_mode,
        in_place=in_place,
    )


@pytest.mark.parametrize("input_spec", parameters["test_run_max_pool_mem_config"]["input_specs"])
@pytest.mark.parametrize("dtype", parameters["test_run_max_pool_mem_config"]["dtype"])
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_run_max_pool_mem_config(device, dtype, input_spec, memory_config):
    (
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec
    run_max_pool2d(
        batch_size,
        input_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_h,
        strid_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        dtype,
        device,
        ceil_mode=ceil_mode,
        memory_config=memory_config,
    )


def adaptive_to_max_pool2d(input_tensor, output_size):
    """
    Convert AdaptiveMaxPool2d to equivalent MaxPool2d operation
    Handles cases where input dimensions are not perfectly divisible by output dimensions

    Args:
        input_tensor: Input tensor of shape (N, C, H, W)
        output_size: Desired output size (tuple or int)

    Returns:
        Tuple of (kernel_size, stride, padding) for MaxPool2d and a note if exact conversion isn't possible
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    input_height, input_width = input_tensor.shape[2], input_tensor.shape[3]
    output_height, output_width = output_size

    # Check if dimensions are valid
    if input_height < output_height or input_width < output_width:
        raise ValueError("Output size cannot be larger than input size for max pooling")

    # Calculate stride (might be floating point)
    stride_h_float = input_height / output_height
    stride_w_float = input_width / output_width

    # Round down stride to integer
    stride_h = math.floor(stride_h_float)
    stride_w = math.floor(stride_w_float)

    # Ensure stride is at least 1
    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)

    # Calculate kernel size
    kernel_h = input_height - (output_height - 1) * stride_h
    kernel_w = input_width - (output_width - 1) * stride_w

    # Handle case where kernel size might be too large
    if kernel_h > input_height:
        kernel_h = input_height
    if kernel_w > input_width:
        kernel_w = input_width

    # Calculate if this is an exact conversion
    is_exact = (
        stride_h_float == stride_h
        and stride_w_float == stride_w
        and input_height == (output_height - 1) * stride_h + kernel_h
        and input_width == (output_width - 1) * stride_w + kernel_w
    )

    message = ""
    if not is_exact:
        message = (
            "Note: This is an approximation. For non-integer stride ratios, "
            "AdaptiveMaxPool2d uses a more complex logic with varying kernel sizes."
        )

    return (kernel_h, kernel_w), (stride_h, stride_w), (0, 0), message


@pytest.mark.parametrize(
    "input_height, input_width, output_height, output_width",
    [
        # [80, 80, 3, 3],
        [20, 20, 3, 3],
        # [40, 40, 3, 3],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_adaptive_to_max_pool2d(device, input_height, input_width, output_height, output_width):
    batch_size = 1
    input_channels = 256
    input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)

    # Define output size
    output_size = (output_height, output_width)

    # Use AdaptiveMaxPool2d
    adaptive_pool = torch.nn.AdaptiveMaxPool2d(output_size)
    adaptive_output = adaptive_pool(input_tensor)

    # Convert to MaxPool2d
    kernel_size, stride, padding, message = adaptive_to_max_pool2d(input_tensor, output_size)
    print(kernel_size)
    print(stride)
    print(padding)
    # max_pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    # max_pool_output = max_pool(input_tensor)

    nhwc_input = input_tensor.permute(0, 2, 3, 1)
    nhwc_input = nhwc_input.reshape(1, 1, batch_size * input_height * input_width, input_channels)
    ttnn_input = ttnn.from_torch(nhwc_input, ttnn.bfloat16, device=device)
    max_pool_output = ttnn.max_pool2d(
        input_tensor=ttnn_input,
        batch_size=batch_size,
        input_h=input_height,
        input_w=input_width,
        channels=input_channels,
        kernel_size=[kernel_size[0], kernel_size[1]],
        stride=[stride[0], stride[1]],
        padding=[padding[0], padding[1]],
        dilation=[1, 1],
        applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    max_pool_output = ttnn.to_torch(max_pool_output)
    max_pool_output = max_pool_output.reshape(batch_size, output_height, output_width, input_channels)
    max_pool_output = max_pool_output.permute(0, 3, 1, 2)

    # Compare outputs
    difference = torch.abs(adaptive_output - max_pool_output).sum().item()
    print(f"Input shape: {input_tensor.shape}")
    print(f"Target output shape: {output_size}")
    print(f"Calculated parameters - kernel_size: {kernel_size}, stride: {stride}, padding: {padding}")
    print(f"AdaptiveMaxPool2d output shape: {adaptive_output.shape}")
    print(f"MaxPool2d output shape: {max_pool_output.shape}")
    print(f"Difference between outputs: {difference}")
    print(f"Are outputs identical? {torch.allclose(adaptive_output, max_pool_output)}")
