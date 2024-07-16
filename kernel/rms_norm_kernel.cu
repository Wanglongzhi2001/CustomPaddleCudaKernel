#include "utils.h"
#include "stdio.h"
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}


#define WARP_SIZE 32

template <int blockSize>
__device__ float blockReduceSum(float sum){
    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[WARP_SIZE]; 
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if(laneId == 0 )warpLevelSums[warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    // Final reduce using first warp
    if (warpId == 0) sum = warpReduceSum<blockSize/WARP_SIZE>(sum); 
    // write result for this block to global mem
    return sum;
}

template<int blockSize>
__global__ void rms_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ norm_weight,
    float* __restrict__ output,
    const float* epsilon,
    const int num_tokens,
    const int hidden_size) {
    
    // 每个 block 的 variance
    __shared__ float s_variance;

    const int tid = threadIdx.x;
    if (tid == 0) printf("you have got 3\n");
    // 平方值，blockReduce 统计
    float variance = 0.0f;

    // 先用网格跨步循环将元素 offload 到一个 block 内的元素
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const int input_idx = blockIdx.x * hidden_size + i;
        variance += input[input_idx] * input[input_idx];
    }

    if (tid == 0) printf("you have got 4\n");
    // block ReduceSum 
    variance = blockReduceSum<blockSize>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / hidden_size + epsilon[0]);
    }
    if (tid == 0) printf("you have got 5\n");

    __syncthreads();
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        const int input_idx = blockIdx.x * hidden_size + i;
        const int output_idx = blockIdx.x * hidden_size + i;
        output[output_idx] = input[input_idx] * s_variance * norm_weight[i];
    }
}

void launchRMSNorm(paddle::Tensor& input, paddle::Tensor& weight, paddle::Tensor& epsilon, paddle::Tensor& output) {
    printf("you have got 1\n");
    int hidden_size = input.dims()[1];
    int num_tokens = input.dims()[0];
    const int blockSize = 1024;
    dim3 grid(num_tokens);
    dim3 block(blockSize);
    auto stream = input.stream();

    printf("you have got 2\n");
    // 一个 block 计算一行数据
    rms_norm_kernel<blockSize><<<grid, block, 0, stream>>>(input.data<float>(), weight.data<float>(), output.data<float>(), epsilon.data<float>(), num_tokens, hidden_size);
}



void MyRMSNorm(paddle::Tensor& input,
            paddle::Tensor& weight,
            paddle::Tensor&  epsilon,
            paddle::Tensor& output) {
    launchRMSNorm(input, weight, epsilon, output);
}

PD_BUILD_OP(my_rms_norm)
    .Inputs({"input", "weight", "epsilon", "output"})
    .Outputs({"Out"})
    .SetInplaceMap({{"output", "Out"}})
    .SetKernelFn(PD_KERNEL(MyRMSNorm));
