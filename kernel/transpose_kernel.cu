#include "utils.h"


// 2D matrix transpose
template<typename T>
__global__ void transpose_kernel_v1(const T* __restrict__ x,
                                T* __restrict__ out,
                                int M,
                                int N) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid_x >= M || tid_y >= N) {
        return;
    }
    // 注意 out 的形状是 [N, M]
    out[tid_y * M + tid_x] = x[tid_x * N + tid_y];
}

// Add shared memory support
// 一个 block 处理一块的数据，块大小可能小于 smem 的大小（一个线程处理多个元素）
// TODO: fix accuracy when M not equal to N
template<typename T>
__global__ void transpose_kernel_v2(const T* __restrict__ x,
                                T* __restrict__ out,
                                int M,
                                int N) {
    __shared__ float smem[32][32];

    const int ITER_X = 32 / blockDim.x;
    const int ITER_Y = 32 / blockDim.y;
    // load data from global memory to smem 
#pragma unroll
    for (int iy = 0; iy < ITER_Y; ++iy) {
        const int global_y = blockDim.y * blockIdx.y + threadIdx.y;
        const int local_y = iy * blockDim.y + threadIdx.y % blockDim.y;
#pragma unroll
        for (int ix = 0; ix < ITER_X; ++ix) {
            const int global_x = blockDim.x * blockIdx.x + threadIdx.x;
            const int local_x = ix * blockDim.x + threadIdx.x % blockDim.x;
            if (global_x < M && global_y < N) {
                // smem[local_x][local_y] = x[global_x * N + global_y];
                smem[local_x][local_y] = x[global_y * N + global_x];
            }
        }
    }
    __syncthreads();

    // store data from smem to output
#pragma unroll
    for (int iy = 0; iy < ITER_Y; ++iy) {
        const int global_y = blockDim.x * blockIdx.x + threadIdx.y;
        const int local_y = iy * blockDim.y + threadIdx.y % blockDim.y;
#pragma unroll
        for (int ix = 0; ix < ITER_X; ++ix) {
            const int global_x = blockDim.y * blockIdx.y + threadIdx.x;
            const int local_x = ix * blockDim.x + threadIdx.x % blockDim.x;
            if (global_x < M && global_y < N) {
                // out[global_y * M + global_x] = smem[local_x][local_y];
                out[global_y * M + global_x] = smem[local_y][local_x];
            }
        }
    }
}


// Deal with bank conflict
template<typename T>
__global__ void transpose_kernel_v3(const T* __restrict__ x,
                                T* __restrict__ out,
                                int M,
                                int N) {
    __shared__ float smem[32][33];

    const int ITER_X = 32 / blockDim.x;
    const int ITER_Y = 32 / blockDim.y;
    // load data from global memory to smem 
#pragma unroll
    for (int iy = 0; iy < ITER_Y; ++iy) {
        const int global_y = blockDim.y * blockIdx.y + threadIdx.y;
        const int local_y = iy * blockDim.y + threadIdx.y % blockDim.y;
#pragma unroll
        for (int ix = 0; ix < ITER_X; ++ix) {
            const int global_x = blockDim.x * blockIdx.x + threadIdx.x;
            const int local_x = ix * blockDim.x + threadIdx.x % blockDim.x;
            if (global_x < M && global_y < N) {
                // smem[local_x][local_y] = x[global_x * N + global_y];
                smem[local_x][local_y] = x[global_y * N + global_x];
            }
        }
    }
    __syncthreads();

    // store data from smem to output
#pragma unroll
    for (int iy = 0; iy < ITER_Y; ++iy) {
        const int global_y = blockDim.x * blockIdx.x + threadIdx.y;
        const int local_y = iy * blockDim.y + threadIdx.y % blockDim.y;
#pragma unroll
        for (int ix = 0; ix < ITER_X; ++ix) {
            const int global_x = blockDim.y * blockIdx.y + threadIdx.x;
            const int local_x = ix * blockDim.x + threadIdx.x % blockDim.x;
            if (global_x < M && global_y < N) {
                // out[global_y * M + global_x] = smem[local_x][local_y];
                out[global_y * M + global_x] = smem[local_y][local_x];
            }
        }
    }
}


std::vector<paddle::Tensor> MyTranspose(const paddle::Tensor& x) {
    int M = x.dims()[0];
    int N = x.dims()[1];
    int dimx = 32;
    int dimy = 32;
    const int n = x.numel();
    dim3 grid((M  + dimx - 1) / dimx, (N + dimy - 1) / dimy);
    dim3 block(dimx , dimy);
    auto stream = x.stream();
    auto out = paddle::full({N, M}, 0, x.dtype(), x.place());
    transpose_kernel_v2<<<grid, block, 0, stream>>>(x.data<float>(), out.data<float>(), M, N);
    return {out};
}

std::vector<std::vector<int64_t>> MyTransposeInferShape(const std::vector<int64_t>& x_shape) {
    return {{x_shape[1], x_shape[0]}};
}

std::vector<paddle::DataType> MyTransposeInferDtype(const paddle::DataType& x_dtype) {
    return {x_dtype};
}

PD_BUILD_OP(my_transpose)
    .Inputs({"x"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(MyTranspose))
    .SetInferShapeFn(PD_INFER_SHAPE(MyTransposeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MyTransposeInferDtype));