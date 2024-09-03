#include "utils.h"

// memory copy
template<typename T>
__global__ void memcpy_kernel(T* __restrict__ out, const T* __restrict__ x, int M, int N) {
  const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int global_tid = tid_y * N + tid_x;

  if (tid_x < N || tid_y < M) {
    out[global_tid] = x[global_tid];
  }
}

// 2D matrix transpose
template<typename T>
__global__ void transpose_kernel_v1(T* __restrict__ out, const T* __restrict__ x, int M, int N) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid_x >= M || tid_y >= N) {
        return;
    }
    // 注意 out 的形状是 [N, M]
    out[tid_y * M + tid_x] = x[tid_x * N + tid_y];
}

// 一个 block 处理多个块（一个线程处理多个元素），用 shared memory 存储每一趟的中间结果
template<typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void transpose_v2(T *odata, const T *idata, int M, int N) {
    const int warpSize = 32;
    __shared__ float smem[warpSize][warpSize];

    constexpr const int ITER_NUM_X = warpSize / BLOCK_DIM_Y;
    constexpr const int ITER_NUM_Y = warpSize / BLOCK_DIM_X;

    // global memory to shared memory
#pragma unroll
    for (int iy = 0; iy < ITER_NUM_Y; iy++) {
#pragma unroll
      for (int ix = 0; ix < ITER_NUM_X; ix++) {
        const int smem_ly = iy * BLOCK_DIM_X + threadIdx.x % BLOCK_DIM_X;
        const int gy = blockIdx.y * warpSize + ix * BLOCK_DIM_Y + threadIdx.y % BLOCK_DIM_Y;

        const int smem_lx = ix * BLOCK_DIM_Y + threadIdx.y % BLOCK_DIM_Y;
        const int gx = blockIdx.x * warpSize + iy * BLOCK_DIM_Y + threadIdx.x % BLOCK_DIM_Y;
        if (gy < M && gx < N) {
          smem[smem_lx][smem_ly] = idata[gy * N + gx];
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < ITER_NUM_Y; iy++) {
#pragma unroll
      for (int ix = 0; ix < ITER_NUM_X; ix++) {
      const int smem_ly = iy * BLOCK_DIM_X + threadIdx.x % BLOCK_DIM_X;;
      const int gy = blockIdx.x * warpSize + ix * BLOCK_DIM_Y + threadIdx.y % BLOCK_DIM_Y;

        const int smem_lx = ix * BLOCK_DIM_Y + threadIdx.y % BLOCK_DIM_Y;
        const int gx = blockIdx.y * warpSize + iy * BLOCK_DIM_Y + threadIdx.x % BLOCK_DIM_Y;
        if (gy < N && gx < M) {
          odata[gy * M + gx] = smem[smem_ly][smem_lx];
        }
      }
    }
}

// 一个 block 处理多个块（一个线程处理多个元素），用 shared memory 存储每一趟的中间结果
template<typename T, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void transpose_v3(T *odata, const T *idata, int M, int N) {
    const int warpSize = 32;
    __shared__ float smem[warpSize][warpSize+1];

    constexpr const int ITER_NUM_X = warpSize / BLOCK_DIM_Y;
    constexpr const int ITER_NUM_Y = warpSize / BLOCK_DIM_X;

    // global memory to shared memory
#pragma unroll
    for (int iy = 0; iy < ITER_NUM_Y; iy++) {
#pragma unroll
      for (int ix = 0; ix < ITER_NUM_X; ix++) {
        const int smem_ly = iy * BLOCK_DIM_X + threadIdx.x % BLOCK_DIM_X;
        const int gy = blockIdx.y * warpSize + ix * BLOCK_DIM_Y + threadIdx.y % BLOCK_DIM_Y;

        const int smem_lx = ix * BLOCK_DIM_Y + threadIdx.y % BLOCK_DIM_Y;
        const int gx = blockIdx.x * warpSize + iy * BLOCK_DIM_Y + threadIdx.x % BLOCK_DIM_Y;
        if (gy < M && gx < N) {
          smem[smem_lx][smem_ly] = idata[gy * N + gx];
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < ITER_NUM_Y; iy++) {
#pragma unroll
      for (int ix = 0; ix < ITER_NUM_X; ix++) {
      const int smem_ly = iy * BLOCK_DIM_X + threadIdx.x % BLOCK_DIM_X;;
      const int gy = blockIdx.x * warpSize + ix * BLOCK_DIM_Y + threadIdx.y % BLOCK_DIM_Y;

        const int smem_lx = ix * BLOCK_DIM_Y + threadIdx.y % BLOCK_DIM_Y;
        const int gx = blockIdx.y * warpSize + iy * BLOCK_DIM_Y + threadIdx.x % BLOCK_DIM_Y;
        if (gy < N && gx < M) {
          odata[gy * M + gx] = smem[smem_ly][smem_lx];
        }
      }
    }
}

std::vector<paddle::Tensor> MyTranspose(const paddle::Tensor& x) {
    int M = x.dims()[0];
    int N = x.dims()[1];
    auto out = paddle::full({N, M}, 0, x.dtype(), x.place());

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    transpose_v3<float, 32, 32><<< grid, block >>>(out.data<float>(), x.data<float>(), M, N);
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