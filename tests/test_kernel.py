import paddle
import numpy as np
import pytest
from my_paddle_ops import my_elementwise_add, my_transpose


# @pytest.mark.parametrize('m', [32, 64, 128])
# @pytest.mark.parametrize('n', [32, 64, 128])
# def test_elementwise_add(m, n):
#     ele_cnt = m * n
#     shape = [m, n]
#     a = paddle.rand(shape).cuda()
#     b = paddle.rand(shape).cuda()

#     my_res = paddle.zeros(shape).cuda()
#     paddle_res = paddle.add(a, b)

#     my_res = my_elementwise_add(a, b, my_res)

#     np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('m', [256])
@pytest.mark.parametrize('n', [256])
def test_transpose(m, n):
    ele_cnt = m * n
    shape = [m, n]
    transposed_shape = [n, m]
    a = paddle.rand(shape).cuda()

    my_res = paddle.zeros(transposed_shape).cuda()
    paddle_res = paddle.transpose(a, [1, 0])

    my_res = my_transpose(a, my_res)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-5, atol=1e-5)


