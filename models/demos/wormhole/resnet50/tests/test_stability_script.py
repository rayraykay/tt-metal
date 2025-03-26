# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import tracemalloc
from models.utility_functions import (
    run_for_wormhole_b0,
)
import time
from loguru import logger
from models.demos.ttnn_resnet.demo.demo import run_resnet_imagenet_inference

# Define the duration for the test in seconds (24 hours)
TEST_DURATION_SECONDS = 60 * 60 * 24


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, iterations",
    ((16, 100),),
)
def test_resnet_stability(batch_size, iterations, imagenet_label_dict, model_location_generator, mesh_device):
    logger.info(f"Running ResNet stability test for {TEST_DURATION_SECONDS} seconds")

    start = time.time()
    iter = 0
    tracemalloc.start()
    with open("memory_leak_log.txt", "w") as f:
        f.write(f"ResNet Stability Test Memory Stats\n\n")

        while True:
            snapshot1 = tracemalloc.take_snapshot()
            iter += 1

            run_resnet_imagenet_inference(
                batch_size, iterations, imagenet_label_dict, model_location_generator, mesh_device
            )

            snapshot2 = tracemalloc.take_snapshot()
            stats = snapshot2.compare_to(snapshot1, "lineno")

            f.write(f"Iteration {iter}\n")
            for stat in stats[:10]:
                f.write(str(stat) + "\n")

            f.write("\n")

            if time.time() - start > TEST_DURATION_SECONDS:
                break

            print(f"Completed iteration {iter}")

            if iter == 1:
                break

        f.write(f"ResNet stability test completed after {iter} iterations\n")

    logger.info(f"ResNet stability test completed after {iter} iterations")
    print(time.time() - start)
