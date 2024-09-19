import paddle
import numpy as np
import pytest
from my_paddle_ops import my_attention


paddle.seed(2024)
np.random.seed(2024)


def attn_ref(query, key, value, mask=None):
    batch = query.shape[0]
    heads = query.shape[2]
    seq_len = query.shape[1]
    head_dim = query.shape[3]
    scale = head_dim**-0.5

    qk_res = paddle.matmul(query, key, transpose_y=True)
    attention = qk_res * scale
    if mask is not None:
        attention = attention + mask
    softmax_result = paddle.nn.functional.softmax(attention, -1)
    result = paddle.matmul(softmax_result, value)
    return result

# need to optimize kernel's smem, now the dh cannot larger than 1020, unless the needed smem is too large
@pytest.mark.parametrize('bsz', [1, 4, 8])
@pytest.mark.parametrize('nh', [2, 4, 8])
@pytest.mark.parametrize('seq_len', [32, 64, 128])
@pytest.mark.parametrize('dh', [128, 256, 512, 1020])
def test_attention(bsz, nh, seq_len, dh):
    query_shape = [bsz, nh, seq_len, dh]
    query = paddle.rand(query_shape).cuda()
    key = paddle.rand(query_shape).cuda()
    value = paddle.rand(query_shape).cuda()
    paddle_res = attn_ref(query, key, value)
    my_res = my_attention(query, key, value)

    np.testing.assert_allclose(my_res.cpu().numpy(), paddle_res.cpu().numpy(), rtol=1e-2, atol=1e-2)
