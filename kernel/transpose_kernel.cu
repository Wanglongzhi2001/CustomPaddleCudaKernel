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
    out[tid_y * M + tid_x] = x[tid_x * N + tid_y];
}

// Add shared memory support
// 一个 block 处理一块的数据，块大小可能小于 smem 的大小（一个线程处理多个元素）
// TODO: fix accuracy
template<typename T>
__global__ void transpose_kernel_v2(const T* __restrict__ x,
                                T* __restrict__ out,
                                int M,
                                int N) {
    const int smem_x = 32;
    const int smem_y = 32;
    __shared__ float smem[smem_x][smem_x];

    const int ITER_X = smem_x / blockDim.x;
    const int ITER_Y = smem_y / blockDim.y;
    // load data from global memory to smem 
#pragma unroll
    for (int ix = 0; ix < ITER_X; ++ix) {
        const int global_x = blockDim.x * blockIdx.x + threadIdx.x;
        const int local_x = ix * blockDim.x + threadIdx.x % blockDim.x;
#pragma unroll
        for (int iy = 0; iy < ITER_Y; ++iy) {
            const int global_y = blockDim.y * blockIdx.y + threadIdx.y;
            const int local_y = iy * blockDim.y + threadIdx.y % blockDim.y;
            if (global_x < M && global_y < N)
                smem[local_x][local_y] = x[global_x * N + global_y];
        }
    }
    __syncthreads();

    // store data from smem to output
#pragma unroll
    for (int ix = 0; ix < ITER_X; ++ix) {
        const int global_x = blockDim.x * blockIdx.x + threadIdx.x;
        const int local_x = ix * blockDim.x + threadIdx.x % blockDim.x;
#pragma unroll
        for (int iy = 0; iy < ITER_Y; ++iy) {
            const int global_y = blockDim.y * blockIdx.y + threadIdx.y;
            const int local_y = iy * blockDim.y + threadIdx.y % blockDim.y;
            if (global_x < M && global_y < N)
                out[global_x * N + global_y] = smem[local_y][local_x];
        }
    }
}


void MyTranspose(const paddle::Tensor& x,
                    paddle::Tensor& out) {
    int M = x.dims()[0];
    int N = x.dims()[1];
    int dimx = 32;
    int dimy = 32;
    const int n = x.numel();
    dim3 grid((M  + dimx - 1) / dimx, (N + dimy - 1) / dimy);
    dim3 block(dimx , dimy);
    auto stream = x.stream();
    transpose_kernel_v1<<<grid, block, 0, stream>>>(x.data<float>(), out.data<float>(), M, N);
}

PD_BUILD_OP(my_transpose)
    .Inputs({"x", "out"})
    .Outputs({"Out"})
    .SetInplaceMap({{"out", "Out"}})
    .SetKernelFn(PD_KERNEL(MyTranspose));