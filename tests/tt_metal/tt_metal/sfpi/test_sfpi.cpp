// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <filesystem>
#include <string>
#include <string_view>
#include <gtest/gtest.h>
#include <cstdio>

static constexpr std::string_view KernelDir = "tests/tt_metal/tt_metal/test_kernels/sfpi/";

static bool runTest(const std::string& path, unsigned baseLen) {
    const char* phase;

    // compile
    phase = "compile";
    // enqueue
    phase = "queue";
    // execute
    phase = "execute";
    // results
    bool pass = true;
    std::printf("%s: %s\n", path.c_str() + baseLen + 1, pass ? "PASSED" : "FAILED");
    return pass;

failed:
    std::printf("%s: %s FAILED\n", path.c_str() + baseLen + 1, phase);
    return false;
}

static bool runTests(std::string& path, unsigned baseLen) {
    bool pass = true;
    std::vector<std::string> files;
    std::vector<std::string> dirs;

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_directory()) {
            dirs.push_back(entry.path().filename());
        } else {
            files.push_back(entry.path().filename());
        }
    }
    std::sort(files.begin(), files.end());
    std::sort(dirs.begin(), dirs.end());

    path.push_back('/');
    for (const auto& file : files) {
        path.append(file);
        pass &= runTest(path, baseLen);
        path.erase(path.size() - file.size());
    }

    for (const auto& dir : dirs) {
        path.append(dir);
        pass &= runTests(path, baseLen);
        path.erase(path.size() - dir.size());
    }
    path.pop_back();

    return pass;
}

static bool runTestsuite() {
    bool pass = true;

    // Setup

    // Iterate
    std::string path(KernelDir);
    // FIXME: how to find this path relative to pwd?
    pass &= runTests(path, path.size());

    // Teardown

    return pass;
}

TEST(SFPI, TensixSFPI) { EXPECT_TRUE(runTestsuite()); }
