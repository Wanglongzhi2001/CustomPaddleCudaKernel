import paddle
import numpy as np
import pytest
from my_paddle_ops import my_elementwise_add

paddle.seed(2024)
np.random.seed(2024)

@pytest.mark.parametrize('m', [32, 64, 128])
@pytest.mark.parametrize('n', [32, 64, 128])
def test_elementwise_add(m, n):
    shape = [m, n]
    a = paddle.rand(shape).cuda()
    b = paddle.rand(shape).cuda()

    paddle_res = paddle.add(a, b)

    my_res = my_elementwise_add(a, b)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-5, atol=1e-5)