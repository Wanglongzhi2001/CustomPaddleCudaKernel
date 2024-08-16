import paddle
import numpy as np
import pytest
from my_paddle_ops import my_sum


@pytest.mark.parametrize('n', [16 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024])
def test_sum(n):
    a_shape = [n]
    a = paddle.rand(a_shape).cuda()
    paddle_res = paddle.sum(a)
    my_res = my_sum(a)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-3, atol=1e-3)