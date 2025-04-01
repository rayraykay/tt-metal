# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from dataclasses import dataclass
from functools import partial
from collections import namedtuple

from pathlib import Path
from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext

import shutil


def safe_copytree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# --- Copy extra files into the package structure ---
# Copy cpp → ttnn/ttnn/cpp
cpp_src = os.path.join("ttnn", "cpp", "ttnn")
cpp_dst = os.path.join("ttnn", "ttnn", "cpp")
safe_copytree(cpp_src, cpp_dst)

# Copy tt_metal → ttnn/ttnn/tt_metal
metal_src = "tt_metal"
metal_dst = os.path.join("ttnn", "ttnn", "tt_metal")
safe_copytree(metal_src, metal_dst)


class EnvVarNotFoundException(Exception):
    pass


def attempt_get_env_var(env_var_name):
    if env_var_name not in os.environ:
        raise EnvVarNotFoundException(f"{env_var_name} is not provided")
    return os.environ[env_var_name]


def get_is_srcdir_build():
    build_dir = CMakeBuild.get_working_dir()
    assert build_dir.is_dir()
    git_dir = build_dir / ".git"
    return git_dir.exists()


def get_arch_name():
    return "any"


def get_metal_local_version_scheme(metal_build_config, version):
    arch_name = metal_build_config.arch_name

    if version.dirty:
        return f"+g{version.node}.{arch_name}"
    else:
        return ""


def get_metal_main_version_scheme(metal_build_config, version):
    is_release_version = version.distance is None or version.distance == 0
    is_dirty = version.dirty
    is_clean_prod_build = (not is_dirty) and is_release_version

    arch_name = metal_build_config.arch_name

    if is_clean_prod_build:
        return version.format_with("{tag}+{arch_name}", arch_name=arch_name)
    elif is_dirty and not is_release_version:
        return version.format_with("{tag}.dev{distance}", arch_name=arch_name)
    elif is_dirty and is_release_version:
        return version.format_with("{tag}", arch_name=arch_name)
    else:
        assert not is_dirty and not is_release_version
        return version.format_with("{tag}.dev{distance}+{arch_name}", arch_name=arch_name)


def get_version(metal_build_config):
    return {
        "version_scheme": partial(get_metal_main_version_scheme, metal_build_config),
        "local_scheme": partial(get_metal_local_version_scheme, metal_build_config),
    }


def get_from_precompiled_dir():
    """Additional option if the precompiled C++ libs are already in-place."""
    precompiled_dir = os.environ.get("TT_FROM_PRECOMPILED_DIR", None)
    return Path(precompiled_dir) if precompiled_dir else None


@dataclass(frozen=True)
class MetalliumBuildConfig:
    arch_name = get_arch_name()
    from_precompiled_dir = get_from_precompiled_dir()


metal_build_config = MetalliumBuildConfig()


class CMakeBuild(build_ext):
    @staticmethod
    def get_build_env():
        return {
            **os.environ.copy(),
            "CXX": "clang++-17",
        }

    @staticmethod
    def get_working_dir():
        working_dir = Path(__file__).parent
        assert working_dir.is_dir()
        return working_dir

    # This should only run when building the wheel. Should not be running for any dev flow
    # Taking advantage of the fact devs run editable pip install -> "pip install -e ."
    def run(self) -> None:
        if self.is_editable_install_():
            assert get_is_srcdir_build(), f"Editable install detected in a non-srcdir environment, aborting"
            return

        build_env = CMakeBuild.get_build_env()
        source_dir = (
            metal_build_config.from_precompiled_dir
            if metal_build_config.from_precompiled_dir
            else CMakeBuild.get_working_dir()
        )
        assert source_dir.is_dir(), f"Source dir {source_dir} seems to not exist"

        if metal_build_config.from_precompiled_dir:
            build_dir = source_dir / "build"
            assert (build_dir / "lib").exists() and (
                source_dir / "runtime"
            ).exists(), "The precompiled option is selected via `TT_FROM_PRECOMPILED` \
            env var. Please place files into `build/lib` and `runtime` folders."
        else:
            build_dir = source_dir / "build_Release"
            # We indirectly set a wheel build for our CMake build by using BUILD_SHARED_LIBS. This does the following things:
            # - Bundles (most) of our libraries into a static library to deal with a potential singleton bug error with tt_cluster (to fix)
            build_script_args = ["--build-static-libs", "--release"]

            subprocess.check_call(["./build_metal.sh", *build_script_args], cwd=source_dir, env=build_env)

        # Some verbose sanity logging to see what files exist in the outputs
        subprocess.check_call(["ls", "-hal"], cwd=source_dir, env=build_env)
        subprocess.check_call(["ls", "-hal", str(build_dir / "lib")], cwd=source_dir, env=build_env)
        subprocess.check_call(["ls", "-hal", "runtime"], cwd=source_dir, env=build_env)

        # Copy needed C++ shared libraries and runtime assets into wheel (sfpi, FW etc)
        dest_ttnn_build_dir = self.build_lib + "/ttnn/build"
        os.makedirs(dest_ttnn_build_dir, exist_ok=True)
        self.copy_tree(build_dir / "lib", dest_ttnn_build_dir + "/lib")
        self.copy_tree(source_dir / "runtime", self.build_lib + "/runtime")

        # Encode ARCH_NAME into package for later use so user doesn't have to provide
        arch_name_file = self.build_lib + "/ttnn/.ARCH_NAME"
        # should probably change to Python calls to write to a file descriptor instead of calling Linux tools
        subprocess.check_call(f"echo {metal_build_config.arch_name} > {arch_name_file}", shell=True)

        # Move built final built _ttnn SO into appropriate location in ttnn Python tree in wheel
        assert len(self.extensions) == 1, f"Detected {len(self.extensions)} extensions, but should be only 1: ttnn"
        ext = list(self.extensions)[0]
        fullname = self.get_ext_fullname(ext.name)
        filename = self.get_ext_filename(fullname)

        build_lib = self.build_lib
        full_lib_path = build_lib + "/" + filename

        dir_path = os.path.dirname(full_lib_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        src = os.path.join(dest_ttnn_build_dir, build_constants_lookup[ext].so_src_location)
        self.copy_file(src, full_lib_path)
        os.remove(src)

    def is_editable_install_(self):
        return self.inplace


packages = find_namespace_packages(where="ttnn", exclude=["cpp", "cpp.*, *.cpp"])

print(("packaging: ", packages))

# Empty sources in order to force extension executions
ttnn_lib_C = Extension("ttnn._ttnn", sources=[])

ext_modules = [ttnn_lib_C]

BuildConstants = namedtuple("BuildConstants", ["so_src_location"])

build_constants_lookup = {
    ttnn_lib_C: BuildConstants(so_src_location="lib/_ttnn.so"),
}


def expand_patterns(patterns):
    """
    Given a list of glob patterns with brace expansion (e.g. `*.{h,hpp}`),
    return a flat list of glob patterns with the braces expanded.

    It does not check if the files exist.
    """
    expanded = []

    for pattern in patterns:
        if "{" in pattern and "}" in pattern:
            pre = pattern[: pattern.find("{")]
            post = pattern[pattern.find("}") + 1 :]
            options = pattern[pattern.find("{") + 1 : pattern.find("}")].split(",")

            for opt in options:
                expanded.append(f"{pre}{opt}{post}")
        else:
            expanded.append(pattern)

    return expanded


ttnn_patterns = [
    "cpp/**/kernels/**/*.{h,hpp,c,cc,cpp}",
    "tt_metal/api/tt-metalium/*.{h,hpp,c,cc,cpp}",
    "tt_metal/hostdevcommon/api/hostdevcommon/*.{h,hpp,c,cc,cpp}",
    "tt_metal/include/compute_kernel_api/*.{h,hpp,c,cc,cpp}",
    "tt_metal/third_party/tt_llk/**/*.{h,hpp,c,cc,cpp}",
    "tt_metal/impl/dispatch/kernels/**/*.{h,hpp,c,cc,cpp}",
    "tt_metal/kernels/**/*.{h,hpp,c,cc,cpp}",
    "tt_metal/tools/profiler/*.{h,hpp,c,cc,cpp}",
    "tt_metal/fabric/mesh_graph_descriptors/*.yaml",
    "tt_metal/core_descriptors/*.yaml",
    "tt_metal/soc_descriptors/*.yaml",
]

ttnn_package_data = expand_patterns(ttnn_patterns)

print(ttnn_package_data)
setup(
    url="http://www.tenstorrent.com",
    use_scm_version=get_version(metal_build_config),
    packages=packages,
    package_dir={
        "": "ttnn",
    },
    package_data={
        "ttnn": ttnn_package_data,
    },
    include_package_data=True,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
