import paddle
import numpy as np
import pytest
from my_paddle_ops import my_qkv_split

paddle.seed(2024)
np.random.seed(2024)

@pytest.mark.parametrize('bsz', [1, 2, 4])
@pytest.mark.parametrize('num_head', [2, 4, 8])
@pytest.mark.parametrize('dim_head', [128, 256, 512])
def test_transpose(bsz, num_head, dim_head):
    q = paddle.rand([bsz, num_head, dim_head]).cuda()
    k = paddle.rand([bsz, num_head, dim_head]).cuda()
    v = paddle.rand([bsz, num_head, dim_head]).cuda()
    qkv = paddle.concat(
        [q.unsqueeze(1), 
        k.unsqueeze(1), 
        v.unsqueeze(1)], 
        axis=1)

    q_out, k_out, v_out = my_qkv_split(qkv)
    np.testing.assert_allclose(q.cpu().numpy(), q_out.cpu().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(k.cpu().numpy(), k_out.cpu().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(v.cpu().numpy(), v_out.cpu().numpy(), rtol=1e-5, atol=1e-5)
