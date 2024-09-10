#include "utils.h"

// qkv: [bsz, 3, num_head, dim_head]
// q_out: [bsz, num_head, dim_head]
// k_out: [bsz, num_head, dim_head]
// v_out: [bsz, num_head, dim_head]
// grid_size: [bsz, num_head]
// block_size: [128]
// 一个线程搬运多个元素
template<typename T>
__global__ void qkv_split_kernel_v1(const T *qkv, T *q_out, T *k_out, T *v_out, const int bsz, const int num_head, const int dim_head) {
    const int batch_id = blockIdx.x;
    const int head_id = blockIdx.y;

    // q_out[] = qkv[batch_id * num_head * dim_head + head_id * dim_head];
    for (int i = threadIdx.x; i < dim_head; i += blockDim.x) {
        q_out[batch_id * num_head * dim_head + head_id * dim_head + i] = qkv[batch_id * 3 * num_head * dim_head + head_id * dim_head + i];
        k_out[batch_id * num_head * dim_head + head_id * dim_head + i] = qkv[batch_id * 3 * num_head * dim_head + num_head * dim_head + head_id * dim_head + i];
        v_out[batch_id * num_head * dim_head + head_id * dim_head + i] = qkv[batch_id * 3 * num_head * dim_head + 2 * num_head * dim_head + head_id * dim_head + i];

    }
}


// 相比 v1, 使用向量化访存
template<typename T, int64_t vec_size>
__global__ void qkv_split_kernel_v2(const T *qkv, T *q_out, T *k_out, T *v_out, const int bsz, const int num_head, const int dim_head) {
    const int batch_id = blockIdx.x;
    const int head_id = blockIdx.y;
    AlignedVector<T, vec_size> q_vec, k_vec, v_vec;

    for (int i = threadIdx.x; i < dim_head; i += blockDim.x * vec_size) {
        const int q_offset = batch_id * 3 * num_head * dim_head + head_id * dim_head + i * vec_size;
        const int k_offset = batch_id * 3 * num_head * dim_head + num_head * dim_head + head_id * dim_head + i * vec_size;
        const int v_offset = batch_id * 3 * num_head * dim_head + 2 * num_head * dim_head + head_id * dim_head + i * vec_size;
        Load<T, vec_size>(qkv + q_offset, &q_vec);
        Load<T, vec_size>(qkv + k_offset, &k_vec);  
        Load<T, vec_size>(qkv + v_offset, &v_vec);  

        const int out_offset = batch_id * num_head * dim_head + head_id * dim_head + i * vec_size;

        Store<T, vec_size>(q_vec, q + out_offset);
        Store<T, vec_size>(k_vec, k + out_offset);
        Store<T, vec_size>(v_vec, v + out_offset);
    }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> LaunchMyQKVSplit(const paddle::Tensor& qkv) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    int bsz = qkv.dims()[0];
    int num_head = qkv.dims()[2];
    int dim_head = qkv.dims()[3];

    auto q_out = paddle::full({bsz, num_head, dim_head}, 0, qkv.dtype(), qkv.place());
    auto k_out = paddle::full({bsz, num_head, dim_head}, 0, qkv.dtype(), qkv.place());
    auto v_out = paddle::full({bsz, num_head, dim_head}, 0, qkv.dtype(), qkv.place());

    constexpr int PackSize = VEC_16B / sizeof(DataType_);
    dim3 block(128);
    dim3 grid(bsz, num_head);
    qkv_split_kernel_v2<DataType_, PackSize><<<grid, block>>>(
        reinterpret_cast<const DataType_*>(qkv.data<data_t>()),
        reinterpret_cast<DataType_*>(q_out.data<data_t>()),
        reinterpret_cast<DataType_*>(k_out.data<data_t>()),
        reinterpret_cast<DataType_*>(v_out.data<data_t>()),
        bsz, 
        num_head, 
        dim_head);
    return {q_out, k_out, v_out};
}

std::vector<paddle::Tensor> MyQKVSplit(const paddle::Tensor& qkv) {
    switch (qkv.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchMyQKVSplit<paddle::DataType::BFLOAT16>(qkv);
        }
        case paddle::DataType::FLOAT16: {
            return LaunchMyQKVSplit<paddle::DataType::FLOAT16>(qkv);
        }
        case paddle::DataType::FLOAT32: {
            return LaunchMyQKVSplit<paddle::DataType::FLOAT32>(qkv);
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");
            break;
        }
    }
}

std::vector<std::vector<int64_t>> MyQKVSplitInferShape(const std::vector<int64_t>& qkv_shape) {
    return {{qkv_shape[0], qkv_shape[2], qkv_shape[3]}, {qkv_shape[0], qkv_shape[2], qkv_shape[3]}, {qkv_shape[0], qkv_shape[2], qkv_shape[3]}};
}

std::vector<paddle::DataType> MyQKVSplitInferDtype(const paddle::DataType& qkv_dtype) {
    return {qkv_dtype, qkv_dtype, qkv_dtype};
}

PD_BUILD_OP(my_qkv_split)
    .Inputs({"qkv"})
    .Outputs({"q_out", "k_out", "v_out"})
    .SetKernelFn(PD_KERNEL(MyQKVSplit))
    .SetInferShapeFn(PD_INFER_SHAPE(MyQKVSplitInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MyQKVSplitInferDtype));