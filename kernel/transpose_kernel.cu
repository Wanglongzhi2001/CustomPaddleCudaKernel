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


#define BLOCK_DIM 32

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.
__global__ void transpose_v3(float *odata, const float *idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

        // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

std::vector<paddle::Tensor> MyTranspose(const paddle::Tensor& x) {
    int M = x.dims()[0];
    int N = x.dims()[1];
    auto out = paddle::full({N, M}, 0, x.dtype(), x.place());

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1)/ BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
    transpose_v3<<< grid, block >>>(out.data<float>(), x.data<float>(), N, M);
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