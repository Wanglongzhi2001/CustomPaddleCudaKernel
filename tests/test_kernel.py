import paddle
import numpy as np
import pytest
from my_paddle_ops import my_elementwise_add, my_transpose, my_gemm, my_sum, my_softmax


@pytest.mark.parametrize('m', [32, 64, 128])
@pytest.mark.parametrize('n', [32, 64, 128])
def test_elementwise_add(m, n):
    ele_cnt = m * n
    shape = [m, n]
    a = paddle.rand(shape).cuda()
    b = paddle.rand(shape).cuda()

    my_res = paddle.zeros(shape).cuda()
    paddle_res = paddle.add(a, b)

    my_res = my_elementwise_add(a, b, my_res)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-5, atol=1e-5)


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


@pytest.mark.parametrize('m', [512])
@pytest.mark.parametrize('n', [512])
@pytest.mark.parametrize('k', [512])
def test_gemm(m, n, k):
    ele_cnt = m * n
    a_shape = [m, k]
    b_shape = [k, n]
    res_shape = [m, n]
    a = paddle.rand(a_shape).cuda()
    b = paddle.rand(b_shape).cuda()

    my_res = paddle.zeros(res_shape).cuda()
    paddle_res = paddle.matmul(a, b)

    my_res = my_gemm(a, b, my_res)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('n', [32 * 1024 * 1024])
def test_sum(n):
    a_shape = [n]
    res_shape = [n]
    a = paddle.rand(a_shape).cuda()
    my_res = paddle.zeros(res_shape).cuda()
    paddle_res = paddle.sum(a)
    my_res = my_sum(a, my_res)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('m', [64])
@pytest.mark.parametrize('n', [1024])
def test_softmax(m, n):
    a_shape = [m, n]
    res_shape = [m, n]
    a = paddle.rand(a_shape).cuda()

    my_res = paddle.zeros(res_shape).cuda()
    paddle_res = paddle.nn.functional.softmax(a)
    my_res = my_softmax(a, my_res)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-3, atol=1e-3)
