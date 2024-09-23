/*
 * Adapted from
 * https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu
 */ 
#include "utils.h"


// query: [bsz, num_head, seq_len, dim_head]
// key: [bsz, num_head, seq_len, dim_head]
// value: [bsz, num_head, seq_len, dim_head]
// attn_out: [bsz, num_head, seq_len, dim_head]
// grid_size: (bsz, num_head)
// block_size: (Bc)
// m: [bsz, nh, seq_len] max(x_i)
// f: exp(x-m_x) softmax的分子
// l: [bsz, nh, seq_len] sum_fx softmax的分母
template <typename T>
__global__ void multiHeadAttentionKernel(const T* Q, const T* K, const T* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const T softmax_scale,
                    T* l, T *m, T* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    // Tc = N / Bc
    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        // Tr = N / Br
        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

std::vector<paddle::Tensor> MyAttention(const paddle::Tensor& Q,
                                        const paddle::Tensor& K,
                                        const paddle::Tensor& V) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 16; const int Br = 16;

    const int B = Q.dims()[0]; const int nh = Q.dims()[1];
    const int N = Q.dims()[2]; const int d = Q.dims()[3];

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = paddle::full(Q.shape(), 0, Q.dtype(), Q.place());
    auto l = paddle::full({B, nh, N}, 0, Q.dtype(), Q.place());
    auto m = paddle::full({B, nh, N}, -INFINITY, Q.dtype(), Q.place());

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sram_size > max_sram_size) {
        PD_THROW("SRAM size required (",sram_size, ") exceeds device limit (", max_sram_size,")");
    }

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    multiHeadAttentionKernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data<float>(), K.data<float>(), V.data<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data<float>(), m.data<float>(), O.data<float>()
    );
    return {O};
}


std::vector<std::vector<int64_t>> MyAttentionInferShape(const std::vector<int64_t>& query_shape,
                                                        const std::vector<int64_t>& key_shape,
                                                        const std::vector<int64_t>& value_shape) {
    return {query_shape};
}

std::vector<paddle::DataType> MyAttentionInferDtype(const paddle::DataType& query_dtype,
                                                    const paddle::DataType& key_dtype,
                                                    const paddle::DataType& value_dtype) {
    return {query_dtype};
}


PD_BUILD_OP(my_attention)
    .Inputs({"query", "key", "value"})
    .Outputs({"attn_out"})
    .SetKernelFn(PD_KERNEL(MyAttention))
    .SetInferShapeFn(PD_INFER_SHAPE(MyAttentionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MyAttentionInferDtype));