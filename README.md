# CustomTorchCudaKernel
实现自定义PyTorch的CUDA算子，用来学习CUDA以及benchmark和profile

# Usage
## build
```
cd pytorch
mkdir build && cd build
cmake ..
make -j$(nproc)
```
## run benchmark
```
cd ../benchmark
python benchmark.py
```