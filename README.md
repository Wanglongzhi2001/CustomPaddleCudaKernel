# CustomPaddleCudaKernel
实现自定义Paddle的CUDA算子，用来学习CUDA以及benchmark和profile

# Usage
Please make sure you have installed gpu-version of paddle!
## build
```
python setup_cuda.py install
```
## run test
### run all tests
```
pytest ./tests
```

### run specific test
For example, test elementwise_add op:
```
pytest ./tests/elementwise_add.py
```

## kernel roadmap
### 二维
- [x] elementwise_add
- [x] reduce_sum
- [x] transpose
- [x] rms_norm
- [x] gemm
- [x] softmax

### 高维
- [x] qkv_split
- [x] attention


