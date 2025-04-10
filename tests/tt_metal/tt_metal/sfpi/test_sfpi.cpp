// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_fixture.hpp"

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

static constexpr std::string_view KernelDir = "tests/tt_metal/tt_metal/test_kernels/sfpi/";

static bool runTest(tt::tt_metal::IDevice* device, const CoreCoord& coord, const std::string& path, unsigned baseLen) {
    const char* phase;

    // FIXME: Should we build all these first, and then run them?
    auto program(tt::tt_metal::CreateProgram());
    auto kernel = CreateKernel(program, path, coord, tt::tt_metal::ComputeConfig{});
    EnqueueProgram(device->command_queue(), program, false);
    Finish(device->command_queue());

    bool pass = true;
    std::printf("%s: %s\n", path.c_str() + baseLen + 1, pass ? "PASSED" : "FAILED");
    return pass;

failed:
    std::printf("%s: %s FAILED\n", path.c_str() + baseLen + 1, phase);
    return false;
}

static bool runTests(
    tt::tt_metal::IDevice* device, const tt::tt_metal::CoreCoord coord, std::string& path, unsigned baseLen) {
    bool pass = true;
    std::vector<std::string> files;
    std::vector<std::string> dirs;

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_directory()) {
            dirs.push_back(entry.path().filename());
        } else if (entry.path().filename().extension() == ".cpp") {
            files.push_back(entry.path().filename());
        }
    }
    std::sort(files.begin(), files.end());
    std::sort(dirs.begin(), dirs.end());

    path.push_back('/');
    for (const auto& file : files) {
        path.append(file);
        pass &= runTest(device, coord, path, baseLen);
        path.erase(path.size() - file.size());
    }

    for (const auto& dir : dirs) {
        path.append(dir);
        pass &= runTests(device, coord, path, baseLen);
        path.erase(path.size() - dir.size());
    }
    path.pop_back();

    return pass;
}

static bool runTestsuite(tt::tt_metal::IDevice* device, const tt::tt_metal::CoreCoord coord) {
    std::string path;
    if (auto* var = std::getenv("TT_METAL_HOME")) {
        path.append(var);
        if (!path.empty()) {
            path.push_back('/');
        }
    }
    path.append(KernelDir);
    return runTests(device, coord, path, path.size());
}

using tt::tt_metal::CommandQueueSingleCardProgramFixture;

TEST_F(CommandQueueSingleCardProgramFixture, TensixSFPI) {
    CoreCoord core{0, 0};
    for (auto* device : devices_) {
        EXPECT_TRUE(runTestsuite(device, core));
    }
}
