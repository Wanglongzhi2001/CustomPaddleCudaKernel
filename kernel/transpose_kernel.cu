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

#define BLOCK_DIM 32

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.
template<typename T>
__global__ void transpose_v2(T *odata, const T *idata, int M, int N)
{
	__shared__ float smem[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < N) && (yIndex < M))
	{
		unsigned int index_in = yIndex * N + xIndex;
		smem[threadIdx.y][threadIdx.x] = idata[index_in];
	}

        // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < M) && (yIndex < N))
	{
		unsigned int index_out = yIndex * M + xIndex;
		odata[index_out] = smem[threadIdx.x][threadIdx.y];
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
      const int ly = iy * BLOCK_DIM_X + threadIdx.x % BLOCK_DIM_X;
      const int gy = blockIdx.y * warpSize + ly;
#pragma unroll
      for (int ix = 0; ix < ITER_NUM_X; ix++) {
        const int lx = ix * BLOCK_DIM_Y + threadIdx.y % BLOCK_DIM_Y;
        const int gx = blockIdx.x * warpSize + lx;
        if (gy < M && gx < N) {
          smem[lx][ly] = idata[gy * N + gx];
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < ITER_NUM_Y; iy++) {
      const int ly = iy * BLOCK_DIM_X + threadIdx.x % BLOCK_DIM_X;;
      const int gy = blockIdx.x * warpSize + ly;
#pragma unroll
      for (int ix = 0; ix < ITER_NUM_X; ix++) {
        const int lx = ix * BLOCK_DIM_Y + threadIdx.y % BLOCK_DIM_Y;
        const int gx = blockIdx.y * warpSize + lx;
        if (gy < N && gx < M) {
          odata[gy * M + gx] = smem[ly][lx];
        }
      }
    }
   
}

std::vector<paddle::Tensor> MyTranspose(const paddle::Tensor& x) {
    int M = x.dims()[0];
    int N = x.dims()[1];
    auto out = paddle::full({N, M}, 0, x.dtype(), x.place());

    // dim3 block(BLOCK_DIM, BLOCK_DIM);
    // dim3 grid((N + BLOCK_DIM - 1)/ BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
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