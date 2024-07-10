# CustomPaddleCudaKernel
实现自定义Paddle的CUDA算子，用来学习CUDA以及benchmark和profile

# Usage
Please make sure you have installed gpu-version of paddle!
## build
```
python setup_cuda.py install
```
## run test
```
pytest ./tests/test_kernel.py
```

## kernel roadmap

- [x] elementwise_add
- [x] reduce_sum
- [x] transpose
- [ ] gemm
- [ ] softmax
- [ ] attention


