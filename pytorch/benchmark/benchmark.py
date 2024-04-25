import torch
import time

LIB_PATH = "../build/libmy_add.so"

if __name__ == '__main__':
    torch.ops.load_library(LIB_PATH)

    # init input
    n = 4096
    a = torch.rand([1,n]).cuda()
    b = torch.rand([1,n]).cuda()
    c = torch.empty(n).to(device="cuda:0")
    warmup_times = 100
    repeat_times = 1000
    # warmup
    for _ in range(warmup_times):
        torch_res = torch.add(a, b)
    # run torch kernel
    tic = time.perf_counter()
    for _ in range(repeat_times):
        torch_res = torch.add(a, b)
    toc = time.perf_counter()
    print(f"run torch kernel cost: {(toc - tic) / repeat_times} s")


    # warmup
    for _ in range(warmup_times):
        torch_res = torch.add(a, b)
    # run torch kernel
    tic = time.perf_counter()
    for _ in range(repeat_times):
        torch.ops.my_add.torch_launch_my_add(c, a, b, n)
    toc = time.perf_counter()
    print(f"run custom kernel cost: {(toc - tic) / repeat_times} s")


