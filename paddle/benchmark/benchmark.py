import paddle
import time
import numpy as np
from my_paddle_ops import my_elementwise_add, my_transpose

def bench_elementwise_add():
    # init input
    m = 4096
    n = 4096
    ele_cnt = m * n
    shape = [m, n]
    a = paddle.rand(shape).cuda()
    b = paddle.rand(shape).cuda()

    my_res = paddle.zeros(shape).cuda()
    paddle_res = paddle.add(a, b)

    my_res = my_elementwise_add(a, b, my_res)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-5, atol=1e-5)


def bench_transpose():
    # init input
    m = 32
    n = 8
    ele_cnt = m * n
    shape = [m, n]
    a = paddle.rand(shape).cuda()

    my_res = paddle.zeros(shape).cuda()
    paddle_res = paddle.transpose(a, [1, 0])

    my_res = my_transpose(a, my_res)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    bench_transpose()


