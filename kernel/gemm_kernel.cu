#include "utils.h"


// 将二维数组的行列索引转成一维数组的行列索引，这样可以更高效访问数据
// row, col：二维数组实际的行列索引，ld表示该数组实际的列数
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// 开 M*N 个线程，一个线程计算 c 的一个元素，一个线程计算 A 的一行和 B 的一列的累加和
__global__ void naiveSgemm_kernel(
    float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
    const int M, const int N, const int K) {
    // 当前thread在C矩阵中的row
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    // 当前thread在C矩阵中的col
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && n < N) {
        float psum = 0.0;
        // 告知编译器自动展开循环体，这样可以减少循环控制的开销（循环次数小的时候可以这么做）
        #pragma unroll
        // 取出A[row]和B[col]，然后逐个元素相乘累加，得到最终结果
        for (int k = 0; k < K; k++) {
            // a[OFFSET(m, k, K)]: 获取A[m][k]
            // b[OFFSET(k, n, N)]: 获取B[k][n]
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}


void launchNaiveGemm(paddle::Tensor& a,
            paddle::Tensor& b,
            paddle::Tensor& c) {
    const int M = a.dims()[0];
    const int K = a.dims()[1];
    const int N = b.dims()[1];


    int dimx = 32;
    int dimy = 32;
    // 注意这里的 block 数目设置是反的，因为 CUDA 是 x 维度在右上，y 维度在左下，和数组相反
    dim3 grid((N  + dimy - 1) / dimy, (M + dimx - 1) / dimx);
    dim3 block(dimy , dimx);
    auto stream = a.stream();
    naiveSgemm_kernel<<<grid, block, 0, stream>>>(a.data<float>(), b.data<float>(), c.data<float>(), M, N, K);
}


// 分块 shared memory + split-K 寄存器 (存在 bank-conflict)
__global__ void sgemm_V1(
    float* __restrict__ a, float* __restrict__ b, float* __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;  // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;  // (tid % 2 == 0) ? 0 : 4, col of s_a
    int load_b_smem_k = tid >> 5;   // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

    int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;   // global col of a
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;   // global row of b
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}

void launchSGemmv1(paddle::Tensor& a,
            paddle::Tensor& b,
            paddle::Tensor& c) {
    const int M = a.dims()[0];
    const int K = a.dims()[1];
    const int N = b.dims()[1];
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    auto stream = a.stream();
    sgemm_V1<<<gridDim, blockDim, 0, stream>>>(a.data<float>(), b.data<float>(), c.data<float>(), M, N, K);
}



void MyGemm(paddle::Tensor& a,
            paddle::Tensor& b,
            paddle::Tensor& c) {
    launchNaiveGemm(a, b, c);
}

PD_BUILD_OP(my_gemm)
    .Inputs({"a", "b", "c"})
    .Outputs({"Out"})
    .SetInplaceMap({{"c", "Out"}})
    .SetKernelFn(PD_KERNEL(MyGemm));