# ncu benchmark 结果
- M: 256, N: 256
- v1: 耗时 6.91 us，计算吞吐 3.31%，内存吞吐 20.66%
- v2: 耗时 8.7 us，计算吞吐 19.64%，内存吞吐 17.92%
- v3: 耗时 7.49 us，计算吞吐 23.13%，内存吞吐 4.36%
ncu 命令：
```
ncu --section regex:'^(?!Nvlink)'  --kernel-name transpose_kernel_v2 -o transpose_kernel_v2 -f pytest ./tests/test_kernel.py
```

# kernel v1
最暴力的写法，开 M * N 个线程，让线程 (x,y) 负责对 A^T 找到其对应的值 A

# kernel v2
kernel v1 的一个问题就是在下面这行代码中 x 的读不连续（相邻线程跨行读取元素），导致没有合并访存。
```
out[tid_y * M + tid_x] = x[tid_x * N + tid_y];
```

因此我们考虑使用 shared memory，shared memory 是没有读写不连续的问题的，随便跨行读取元素。<br>
在 kernel v2 中我们选择分块存入 shared memory，让 shared memory 大小为 [32, 32]，而 block size 也为 [32, 32]，
一个 block 处理一块的数据（由于这里 block 大小和 shared memory 大小相等，所以一个线程处理一个元素）。<br>

这里需要注意的点是输入的读和输出的写的 index 都是 `x[global_y * N + global_x]`和`out[global_y * M + global_x]` 行读写连续的，这里和 v1 的写法不一致，并且这样写会导致 global_index 和 local_index 的映射变得更晦涩，但是这是没办法的， shared memory 可以读写不连续但是这俩还是需要读写连续的。
# kernel v3
kernel v2 的问题是存在 bank conflict，一个简单的方法验证 kernel 是否存在 bank conflict 的方法是在 ncu 界面的左上角点击 Profile 选项 -> 点击 Metric Details 选项，在弹出的界面的搜索框里输入 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` 回车，可以看到下面的 value 为 0 表示无 bank conflict，不为 0 表示有 bank conflict。

![](../assets/transpose_kernel/bank_conflict_metrics.png)

我们解决 bank conflict 的方法也很简单，就是 padding，将 shared memory 的形状改为 [32, 33] 即可。

