import paddle
import numpy as np
import pytest
from my_paddle_ops import my_gemm

paddle.seed(2024)
np.random.seed(2024)

@pytest.mark.parametrize('m', [512])
@pytest.mark.parametrize('n', [512])
@pytest.mark.parametrize('k', [512])
def test_gemm(m, n, k):
    ele_cnt = m * n
    a_shape = [m, k]
    b_shape = [k, n]
    a = paddle.rand(a_shape).cuda()
    b = paddle.rand(b_shape).cuda()

    paddle_res = paddle.matmul(a, b)

    my_res = my_gemm(a, b)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-3, atol=1e-3)