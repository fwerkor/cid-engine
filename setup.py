import os
import sysconfig
from pathlib import Path

import torch
from setuptools import Extension, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
    include_paths,
)


class CIDBuildExtension(BuildExtension):
    def build_extension(self, ext):
        if ext.name not in {
            "cid_engine._C_cann_device",
            "cid_engine._C_cann_fast",
        }:
            return super().build_extension(ext)

        import torch_npu

        ascend_home = Path(
            os.environ.get(
                "ASCEND_HOME_PATH",
                "/usr/local/Ascend/ascend-toolkit/latest",
            )
        )
        torch_root = Path(torch.__file__).resolve().parent
        torch_npu_root = Path(torch_npu.__file__).resolve().parent
        output = Path(self.get_ext_fullpath(ext.name))
        output.parent.mkdir(parents=True, exist_ok=True)

        if ext.name == "cid_engine._C_cann_device":
            bisheng = ascend_home / "compiler" / "ccec_compiler" / "bin" / "bisheng"
            if not bisheng.is_file():
                raise RuntimeError(f"CANN fast-kernel compiler not found: {bisheng}")
            cann_includes = [
                ascend_home / "aarch64-linux" / "ascendc" / "include",
                ascend_home / "aarch64-linux" / "ascendc" / "include" / "basic_api",
                ascend_home / "aarch64-linux" / "tikcpp" / "tikcfw",
            ]
            command = [
                str(bisheng),
                "-x",
                "asc",
                f"--npu-arch={os.environ.get('CID_CANN_ARCH', 'dav-2201')}",
                "-shared",
                "-fPIC",
                "-O3",
                "-std=c++17",
                "csrc/cann/fast_kernels_device.cpp",
                "-o",
                str(output),
            ]
            command.extend(f"-I{path}" for path in cann_includes)
            self.spawn(command)
            return

        device_library = Path(
            self.get_ext_fullpath("cid_engine._C_cann_device")
        )
        if not device_library.is_file():
            raise RuntimeError(
                f"CANN device-kernel library was not built: {device_library}"
            )

        torch_includes = [
            *map(Path, include_paths()),
            Path(sysconfig.get_paths()["include"]),
            torch_npu_root / "include",
            torch_npu_root / "include" / "third_party" / "acl" / "inc",
        ]
        temp_dir = Path(self.build_temp) / "cann_fast"
        temp_dir.mkdir(parents=True, exist_ok=True)
        host_object = temp_dir / "fast_kernels_host.o"
        abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
        cxx = os.environ.get("CXX", "g++")

        host_command = [
            cxx,
            "-c",
            "-fPIC",
            "-O3",
            "-std=c++20",
            f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
            "csrc/cann/fast_kernels_host.cpp",
            "-o",
            str(host_object),
        ]
        host_command.extend(f"-I{path}" for path in torch_includes)
        self.spawn(host_command)

        link_command = [
            cxx,
            "-shared",
            str(host_object),
            "-o",
            str(output),
            f"-L{device_library.parent}",
            "-Wl,-rpath,$ORIGIN",
            "-Wl,--no-as-needed",
            f"-l:{device_library.name}",
            f"-L{torch_root / 'lib'}",
            f"-L{torch_npu_root / 'lib'}",
            "-ltorch_npu",
            "-ltorch",
            "-ltorch_cpu",
            "-lc10",
        ]
        self.spawn(link_command)


with_cuda = torch.version.cuda is not None and CUDA_HOME is not None
with_cann = False
NpuExtension = None
if not with_cuda:
    try:
        import torch_npu  # noqa: F401
        from torch_npu.utils.cpp_extension import NpuExtension

        with_cann = hasattr(torch, "npu")
    except ImportError:
        pass

sources = [
    "csrc/ops.cpp",
    "csrc/cpu/display_stats.cpp",
    "csrc/refine.cpp",
    "csrc/registration.cpp",
    "csrc/bindings.cpp",
]
if with_cuda:
    sources.extend(
        [
            "csrc/cuda/display_stats.cu",
            "csrc/cuda/prefix_allocation.cu",
            "csrc/cuda/materialize_snapshot.cu",
            "csrc/cuda/linear_assignment.cu",
            "csrc/cuda/masked_diffusion.cu",
            "csrc/cuda/display_corruption.cu",
            "csrc/cuda/thought_corruption.cu",
            "csrc/cuda/rollout_state.cu",
        ]
    )
elif with_cann:
    sources.extend(
        [
            "csrc/cann/registration.cpp",
            "csrc/cann/display_stats.cpp",
            "csrc/cann/materialize_snapshot.cpp",
        ]
    )

# PyPI publishes an sdist, so the native extension is compiled on the target
# host. Keep the CPU path vectorized and threaded even when CUDA/CANN support
# is built into the same extension.
compile_args = {"cxx": ["-O3", "-std=c++20", "-march=native", "-fopenmp"]}
link_args = ["-fopenmp"]
if with_cuda:
    compile_args["cxx"].append("-DCID_ENGINE_WITH_CUDA=1")
    compile_args["nvcc"] = [
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
    ]
elif with_cann:
    compile_args["cxx"].append("-DCID_ENGINE_WITH_CANN=1")

if with_cuda:
    extension = CUDAExtension
elif with_cann:
    assert NpuExtension is not None
    extension = NpuExtension
else:
    extension = CppExtension

ext_modules = [
    extension(
        "cid_engine._C",
        sources=sources,
        include_dirs=["csrc/include"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]
if with_cann:
    ext_modules.extend(
        [
            Extension(
                "cid_engine._C_cann_device",
                sources=["csrc/cann/fast_kernels_device.cpp"],
                language="asc",
            ),
            Extension(
                "cid_engine._C_cann_fast",
                sources=["csrc/cann/fast_kernels_host.cpp"],
                language="c++",
            ),
        ]
    )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CIDBuildExtension.with_options(use_ninja=False)},
)
