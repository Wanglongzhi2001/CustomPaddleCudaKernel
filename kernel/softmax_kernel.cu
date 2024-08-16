#include "utils.h"
#include "stdio.h"
#define WARP_SIZE 32

__device__ __forceinline__ float warpReduceSum(float sum, int blockSize) {
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}


// 一个 block 计算一行的数据，一个线程处理一个数据
__global__ void softmax_v1(
                        const float* __restrict__ input,
                        float* __restrict__ output,
                        const int total_elem_cnt,
                        float* block_sum,
                        int blockSize) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    float exp_val = expf(input[idx]);
    // 先做 warp 级别的 blockReduce
    float sum = warpReduceSum(exp_val, blockSize);
    // 再使用 total 汇总所有 warp 的 sum 得到一行总的 sum
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    static __shared__ float warpLevelSums[128];
    if(lane_id == 0) warpLevelSums[warp_id] = sum;
    __syncthreads();
    sum = (threadIdx.x < blockSize / WARP_SIZE) ? warpLevelSums[lane_id] : 0;
    if (warp_id == 0) sum = warpReduceSum(sum, blockSize/WARP_SIZE); 
    if (tid == 0) block_sum[blockIdx.x] = sum;
    if (idx < total_elem_cnt) output[idx] = exp_val / block_sum[blockIdx.x]; 
}

void launchSoftmaxV1(paddle::Tensor& x, paddle::Tensor& output) {
    int M = x.dims()[0];
    int N = x.dims()[1];
    int elem_cnt = x.numel();
    const int BLOCK_SIZE = N;
    int block_num = M;
    dim3 Grid( block_num, 1);
    dim3 Block( BLOCK_SIZE, 1);
    auto stream = x.stream();
    float *block_sum;
    cudaMalloc((void **)&block_sum, M * sizeof(float));
    softmax_v1<<<Grid, Block>>>(x.data<float>(), output.data<float>(), elem_cnt, block_sum, BLOCK_SIZE);
}


std::vector<paddle::Tensor> MySoftmax(paddle::Tensor& x) {
    auto output = paddle::full(x.shape(), 0, x.dtype(), x.place());
    launchSoftmaxV1(x, output);
    return {output};
}

std::vector<std::vector<int64_t>> MySoftmaxInferShape(const std::vector<int64_t>& x_shape) {
    return {x_shape};
}

std::vector<paddle::DataType> MySoftmaxInferDtype(const paddle::DataType& x_dtype) {
    return {x_dtype};
}


PD_BUILD_OP(my_softmax)
    .Inputs({"x"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(MySoftmax))
    .SetInferShapeFn(PD_INFER_SHAPE(MySoftmaxInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MySoftmaxInferDtype));
