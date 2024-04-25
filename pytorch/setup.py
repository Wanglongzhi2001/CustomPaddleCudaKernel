from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_add",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "my_add",
            ["pytorch/my_add_ops.cpp", "kernel/my_add_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)