import paddle
import numpy as np
import pytest
from my_paddle_ops import my_softmax


@pytest.mark.parametrize('m', [64])
@pytest.mark.parametrize('n', [1024])
def test_softmax(m, n):
    a_shape = [m, n]
    a = paddle.rand(a_shape).cuda()

    paddle_res = paddle.nn.functional.softmax(a)
    my_res = my_softmax(a)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-3, atol=1e-3)
