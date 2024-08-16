import paddle
import numpy as np
import pytest
from my_paddle_ops import my_rms_norm


@pytest.mark.parametrize('num_tokens', [7, 83, 256])
@pytest.mark.parametrize('hidden_size', [1768, 1769, 1770, 1771, 5120, 5124, 5125, 5126, 8192])
def test_rms_norm_kernel(num_tokens, hidden_size):
    paddle.device.set_device('gpu')
    x_shape = [num_tokens, hidden_size]
    weight_shape = [hidden_size]
    paddle_x = paddle.randn(shape=x_shape)
    paddle_weight = paddle.randn(shape=weight_shape)
    paddle_bias = paddle.zeros(shape=weight_shape)
    epsilon = 1e-6
    my_epsilon = paddle.to_tensor([1e-6], dtype=paddle.float32)
    paddle_rmsnorm = paddle.incubate.nn.functional.fused_rms_norm(paddle_x, paddle_weight, paddle_bias, epsilon, 1)[0]
    my_res = my_rms_norm(paddle_x, paddle_weight, my_epsilon)
    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_rmsnorm.cpu().numpy(), rtol=1e-5, atol=1e-5)
    
