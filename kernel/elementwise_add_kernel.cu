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


void MyElementWiseAdd(const paddle::Tensor& a,
                    const paddle::Tensor& b,
                    paddle::Tensor& c) {
    int n = a.numel();
    dim3 grid((n  + 256 - 1) / 256);
    dim3 block(256);
    auto stream = a.stream();
    elementwise_add_kernel<float, 4><<<grid, block, 0, stream>>>(c.data<float>(), a.data<float>(), b.data<float>(), n);
}

PD_BUILD_OP(my_elementwise_add)
    .Inputs({"a", "b", "c"})
    .Outputs({"c_out"})
    .SetInplaceMap({{"c", "c_out"}})
    .SetKernelFn(PD_KERNEL(MyElementWiseAdd));