import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

with_cuda = torch.version.cuda is not None and CUDA_HOME is not None

sources = [
    "csrc/ops.cpp",
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
        ]
    )

compile_args = {"cxx": ["-O3", "-std=c++20"]}
if with_cuda:
    compile_args["cxx"].append("-DCID_ENGINE_WITH_CUDA=1")
    compile_args["nvcc"] = [
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
    ]

extension = CUDAExtension if with_cuda else CppExtension

setup(
    ext_modules=[
        extension(
            "cid_engine._C",
            sources=sources,
            include_dirs=["csrc/include"],
            extra_compile_args=compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
