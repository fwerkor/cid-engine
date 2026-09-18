from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    ext_modules=[
        CppExtension(
            "cid_engine._C",
            sources=[
                "csrc/ops.cpp",
                "csrc/registration.cpp",
                "csrc/bindings.cpp",
            ],
            include_dirs=["csrc/include"],
            extra_compile_args={"cxx": ["-O3", "-std=c++20"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
