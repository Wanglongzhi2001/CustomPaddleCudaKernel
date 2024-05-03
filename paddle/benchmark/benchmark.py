import paddle
import time
import numpy as np
from my_paddle_ops import my_elementwise_add

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
    print("paddle_res: ", paddle_res.cpu().numpy())

    my_res = my_elementwise_add(a, b, my_res)

    print("a: ", a.cpu().numpy())
    print("b: ", b.cpu().numpy())

    print("my_res: ", my_res.cpu().numpy())

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-5, atol=1e-5)


def run_elementwise_add():
    # init input
    m = 4096
    n = 4096
    a = paddle.rand([m,n]).cuda()
    b = paddle.rand([m,n]).cuda()
    c = paddle.zeros(n).to(device="cuda:0")
    my_elementwise_add(a, b, c)
    print(my_elementwise_add.cpu().numpy())


if __name__ == '__main__':
    run_elementwise_add()


