#include <torch/extension.h>
#include "my_add.h"

void torch_launch_my_add(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n) {
    launch_my_add((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_my_add",
          &torch_launch_my_add,
          "my_add kernel warpper");
}

TORCH_LIBRARY(my_add, m) {
    m.def("torch_launch_my_add", torch_launch_my_add);
}