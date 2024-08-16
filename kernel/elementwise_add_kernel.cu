#include "utils.h"

template<typename T, int64_t vec_size>
__global__ void elementwise_add_kernel(T* c,
                            const T* a,
                            const T* b,
                            int n) {
    int global_tid = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
    int stride = gridDim.x * blockDim.x * vec_size;

    AlignedVector<T, vec_size> a_vec, b_vec, c_vec;


    // vectorized load
    for (; global_tid < n; global_tid += stride) {
        Load<T, vec_size>(a + global_tid, &a_vec);
        Load<T, vec_size>(b + global_tid, &b_vec);  

        
#pragma unroll
        for (int i = 0; i < vec_size; ++i) {
            c_vec[i] = a_vec[i] + b_vec[i];
        }
        // vectorized store
        Store<T, vec_size>(c_vec, c + global_tid);
    }
}


std::vector<paddle::Tensor> MyElementWiseAdd(const paddle::Tensor& a,
                    const paddle::Tensor& b) {
    int n = a.numel();
    const int vec_size = 4;
    dim3 grid((n  + 256 - 1) / (256 * vec_size));
    dim3 block(256);
    auto stream = a.stream();
    // std::vector<int64_t> c_shape;
    // for (int i = 0; i < a.dims().size(); ++i) {
    //     c_shape.push_back(a.dims()[i]);
    // }
    // std::vector<int64_t> c_shape = a.shape;
    auto c = paddle::full(a.shape(), 0, a.dtype(), a.place());
    elementwise_add_kernel<float, 4><<<grid, block, 0, stream>>>(c.data<float>(), a.data<float>(), b.data<float>(), n);
    return {c};
}

std::vector<std::vector<int64_t>> MyElementWiseAddInferShape(const std::vector<int64_t>& a_shape,
                                                             const std::vector<int64_t>& b_shape) {
    int n = a_shape.size();
    std::vector<std::vector<int64_t>> res_shape;
    for (int i = 0; i < n; ++i) {
        res_shape.push_back({a_shape[i]});
    }
    return res_shape;
}

std::vector<paddle::DataType> MyElementWiseAddInferDtype(const paddle::DataType& a_dtype,
                                                         const paddle::DataType& b_dtype) {
    return {a_dtype, b_dtype};
}


PD_BUILD_OP(my_elementwise_add)
    .Inputs({"a", "b"})
    .Outputs({"c"})
    .SetKernelFn(PD_KERNEL(MyElementWiseAdd))
    .SetInferShapeFn(PD_INFER_SHAPE(MyElementWiseAddInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MyElementWiseAddInferDtype));