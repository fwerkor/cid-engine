import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

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

compile_args = {"cxx": ["-O3", "-std=c++20"]}
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
