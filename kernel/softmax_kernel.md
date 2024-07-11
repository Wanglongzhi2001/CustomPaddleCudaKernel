
# softmax_v1 kernel
线程块切分：

一个 block 负责一行数据，一个线程处理一个元素。

softmax（非数值稳定）的思路就是求出每行元素的 sum。这个 kernel 里面我们一个 block 处理一行数据，对每一个 block 进行前面 reduce_kernel 里一样思路的 blockSum 求出每个 block 的 sum，然后用每个元素除以对应的 sum 即可。需要注意的是因为我目前还不知道有没有没有 block 间通信的 API，只能把每个 block 的 sum 存放到一个数组里给每一行计算 softmax 时使用。

TODO：n < 1024 时会出现 inf，待解决。以及 ncu profile 不到这个 kernel，待解决。