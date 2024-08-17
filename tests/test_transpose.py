import paddle
import numpy as np
import pytest
from my_paddle_ops import my_transpose

paddle.seed(2024)
np.random.seed(2024)

@pytest.mark.parametrize('m', [256])
@pytest.mark.parametrize('n', [512])
def test_transpose(m, n):
    shape = [m, n]
    a = paddle.rand(shape).cuda()
    paddle_res = paddle.transpose(a, [1, 0])
    my_res = my_transpose(a)
    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-5, atol=1e-5)