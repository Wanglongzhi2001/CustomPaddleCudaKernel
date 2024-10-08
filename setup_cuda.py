# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup


def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def get_gencode_flags():
    if not strtobool(os.getenv("FLAG_LLM_PDC", "False")):
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]
    else:
        # support more cuda archs
        return [
            "-gencode",
            "arch=compute_80,code=sm_80",
        ]


gencode_flags = get_gencode_flags()

setup(
    name="my_paddle_ops",
    ext_modules=CUDAExtension(
        sources=[
            "./kernel/elementwise_add_kernel.cu",
            "./kernel/transpose_kernel.cu",
            "./kernel/gemm_kernel.cu",
            "./kernel/reduce_kernel.cu",
            "./kernel/softmax_kernel.cu",
            "./kernel/rms_norm_kernel.cu",
            "./kernel/qkv_split_kernel.cu",
            "./kernel/multi_head_attention_kernel.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            ]
            + gencode_flags,
        },
    ),
)