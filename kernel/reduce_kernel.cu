#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}


template <int blockSize>
__global__ void blockReduceSum(float *g_idata, float *g_odata, const int NUM_PER_THREAD){
    float sum = 0;

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    #pragma unroll
    for(int iter=0; iter<NUM_PER_THREAD; iter++){
        sum += g_idata[i+iter*blockSize];
    }
    
    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[WARP_SIZE]; 
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if(laneId == 0) warpLevelSums[warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0) sum = warpReduceSum<blockSize/WARP_SIZE>(sum); 
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sum;
}


void launchBlockReduce(paddle::Tensor& x, paddle::Tensor& res) {
    int N = x.numel();
    const int BLOCK_SIZE = 256;
    int block_num = 1024;
    int NUM_PER_BLOCK = N / block_num;
    int NUM_PER_THREAD = NUM_PER_BLOCK / BLOCK_SIZE;
    dim3 Grid( block_num, 1);
    dim3 Block( BLOCK_SIZE, 1);
    auto blockSum = paddle::full({block_num}, 0, x.dtype(), x.place());
    blockReduceSum<BLOCK_SIZE><<<Grid,Block>>>(x.data<float>(), blockSum.data<float>(), NUM_PER_THREAD);
 
    // final blockReduceSum
    const int FINAL_BLOCK_SIZE = 256;
    NUM_PER_THREAD = block_num / FINAL_BLOCK_SIZE;
    blockReduceSum<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(blockSum.data<float>(), res.data<float>(), NUM_PER_THREAD);
}



std::vector<paddle::Tensor> MySum(paddle::Tensor& x) {
    auto res = paddle::full({1}, 0, x.dtype(), x.place());
    launchBlockReduce(x, res);
    return {res};
}


std::vector<std::vector<int64_t>> MySumInferShape(const std::vector<int64_t>& x_shape) {
    return {{1}};
}

std::vector<paddle::DataType> MySumInferDtype(const paddle::DataType& x_dtype) {
    return {x_dtype};
}



PD_BUILD_OP(my_sum)
    .Inputs({"x"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(MySum))
    .SetInferShapeFn(PD_INFER_SHAPE(MySumInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MySumInferDtype));
