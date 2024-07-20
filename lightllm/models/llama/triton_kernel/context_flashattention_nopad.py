import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F

TESLA = "Tesla" in torch.cuda.get_device_name(0)


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,  # B_LOC 内部记录每个batch 输入的真实位置， B_SEQ_len 记录当前输入的真实长度
    Out,
    Req_to_tokens,
    B_req_idx,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    kv_group_num,
    b_prompt_cache_len,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum((start_m + 1) * BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n),
            mask=(start_n + offs_n) < block_end_loc,
            other=0,
        )
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :], qk, float("-100000000.0"))

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
        acc = acc * acc_scale[:, None]
        # update acc
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return


@torch.no_grad()
def context_attention_fwd(
    q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs
):
    BLOCK = 128 if not TESLA else 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    sm_scale = 1.0 / (Lq ** 0.5)  # 计算scale系数
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,

    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        req_to_token_indexs,
        b_req_idx,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        kv_group_num=kv_group_num,
        b_prompt_cache_len=b_prompt_cache_len,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_no_prompt_cache(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,  # B_LOC 内部记录每个batch 输入的真实位置， B_SEQ_len 记录当前输入的真实长度
    Out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    kv_group_num,
    BLOCK_M: tl.constexpr,  # 对应flash-atten 伪代码中的 B_r
    BLOCK_DMODEL: tl.constexpr,  # 对应 head dim
    BLOCK_N: tl.constexpr,  # 对应 flash-atten 伪代码中的 B_c
):
    cur_batch = tl.program_id(0)  # 当前所在处理的 batch id, 即当前所在处理的 req index
    cur_head = tl.program_id(1)  # 当前在处理的 head id
    start_m = tl.program_id(2)  # 当前在处理行的哪一段 token, 是一个 index, 对应处理的行范围是 [start_m * BLOCK, start_m * BLOCK + BLOCK)
    # 上面这个 m, 即指行范围怎么切的 block

    cur_kv_head = cur_head // kv_group_num  # 根据 kvgroup num, 得到当前应该使用哪段 kv

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)  # load 当前 req 的 seqlen
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)  # load 当前 req 的 start_loc

    block_start_loc = BLOCK_M * start_m  # 如上面注释所说, 处理的行范围, 左界是 BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)  # 一个 value = range(0,BLOCK_N) 的 1d-tensor
    offs_d = tl.arange(0, BLOCK_DMODEL)  # 一个 value = range(0, BLOCK_DMODEL=head_dim) 的 1d-tensor, 是 d 这一维的 offset
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 一个表示正在处理的行的 index 的 1d-tensor
    off_q = (  # 表示需要处理的 q 矩阵的 *offset*(所谓 offset 就是 index * stride), shape 显然为 [BLOCK_M, 1(可忽略), BLOCK_DMODEL(head_dim)]
        # q 的第一维是 nopad 状态, 要定位当前 req 的 q, 首先通过 cur_batch_in_all_start_index 找到当前 req 的 q 的起始位置
        # 然后再加上 offs_m, 得到一个 [BLOCK_M, 1] 的 1d-tensor, 表示 bs 范围内的 stride
        # 后面通过 + offs_d 再广播到 [BLOCK_M, BLOCK_DMODEL]
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    # k,v 同理, off_k 这里计算时顺便完成了转置, 维度变成了 [BLOCK_DMODEL(head_dim), BLOCK_N]
    # 注意这里还没有增加 cur_batch_in_all_start_index 的部分, 后面需要补上
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

    # 提取 q, 注意这里 tl.load 有一个 mask, offs_m 是按 max_input_len 建立的, 但每个 req input_len 不一, 需要 mask 掉越界的部分
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # 对应 flash-atten 中 m, 即维护的 max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # 对应 flash-atten 中 l, 即维护的 softmax 的分母
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)  # 即临时维护的 o

    # 用 tl.where 来做 if.. 这样会更高效嘛..? 总之同样是用来处理 block 按照 max_input_len 划分而导致的可能的越界问题
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    # n 这个维度其实是 k 转置后的 seqlen 这一维, 它本质上就是 m 维度, 最多 BLOCK_N 稍有不同
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            # load 之前, 需要补上 cur_batch_in_all_start_index 以及 当前 block 的 start_n
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
            other=0.0,
        )
        # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))  # casual mask

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])  # 此时的 p 使用 m_ij
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]  # 刚才的 p 用的 m_ij, 需要调整到 m_i_new, 然后 / l_i_new, 这样后面就可以直接 * 到 v 上了
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]  # 这里是对过去的 acc 做缩放, 分子缩放即 alpha(m_i -> m_i_new), 分子缩放即 l_i -> l_i_new
        # update acc
        v = tl.load(  # 读取要跑的 v
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
            other=0.0,
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)  # acc += p * v
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (  # 获取要写的位置, 结构上和 q 完全一致
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return


@torch.no_grad()
def context_attention_fwd_no_prompt_cache(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        b_start_loc,
        b_seq_len,
        max_input_len):
    """执行 context_attention_fwd

    Args:
        q (torch.Tensor): q, shape = [seqlen(若干个 req 的 token 拼接而成), tp_head_num, tp_head_dim]
        k (torch.Tensor): k, shape = [seqlen(若干个 req 的 token 拼接而成), tp_head_num, tp_head_dim]
        v (torch.Tensor): v, shape = [seqlen(若干个 req 的 token 拼接而成), tp_head_num, tp_head_dim]
        o (torch.Tensor): o buffer, shape = [seqlen(若干个 req 的 token 拼接而成), tp_head_num, tp_head_dim]
        b_start_loc: 记录每个 req 在 seqlen 这维中的 start loc
        b_seq_len: 记录每个 req 的 seqlen, prefill 情况下即 input_ids 长度, pause 的 req 的 input_ids 会长一些(即除了 prompt ids 外还有推的 id)
        max_input_len: 每个 req 这次要推理的 token 数量的最大值(在 no_prompt_cache 情况下, 即为 input_ids 的最大值; 因为是 no_prompt_cache,
        所以如果是 pause 的 req, 也是直接重推所有 input_token_ids)
    """
    BLOCK = 128 if not TESLA else 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]  # 得到 qkv 的 head_dim
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128, 256}

    sm_scale = 1.0 / (Lq ** 0.5)  # 计算scale系数
    batch, head = b_seq_len.shape[0], q.shape[1]  # batch 即 req 数量, head 即 head 个数
    kv_group_num = q.shape[1] // k.shape[1]  # 对应 GQA, MQA 的实现

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # 按照 batch(即 不同 req), head, seqlen 切片 三个维度做无依赖并行
    # grid 会对应 triton kernel 里的 program_id, 从而使你知道你在哪个处理哪个 item
    # 第一维很自然(req 之间天然隔离), 第二维就是 mha 带来的, head 之间天然隔离, 第三维就是对应 flash-atten 的外循环了, flash-atten2
    # 的外循环是行尺度的, 因此第三维是 seqlen.
    # 即除了前面两维外, 最后一维的设计来自 flash-atten

    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel_no_prompt_cache[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return


def torch_att(xq, xk, xv, bs, seqlen, num_head, head_dim, prompt_cache_len):
    xq = xq.view(bs, seqlen, num_head, head_dim)
    xk = xk.view(bs, seqlen + prompt_cache_len, num_head, head_dim)
    xv = xv.view(bs, seqlen + prompt_cache_len, num_head, head_dim)
    mask_cache = torch.ones((seqlen, prompt_cache_len)).cuda().unsqueeze(0).unsqueeze(0).cuda()
    mask = torch.tril(torch.ones(seqlen, seqlen), diagonal=0).unsqueeze(0).unsqueeze(0).cuda()
    mask[mask == 0.0] = -100000000.0
    mask = torch.cat([mask_cache, mask], dim=-1)
    mask = mask.repeat(bs, num_head, 1, 1)
    keys = xk
    values = xv
    xq = xq.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    scores = F.softmax(scores.float() + mask, dim=-1).type_as(xq)
    output = torch.matmul(scores, values).transpose(1, 2).contiguous().reshape(-1, num_head, head_dim)
    return output


def test():
    import torch
    import numpy as np

    Z, H, N_CTX, D_HEAD = 1, 6, 500, 128
    dtype = torch.float16
    Z = 1
    q = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((Z * N_CTX + 7000, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2)
    v = torch.empty((Z * N_CTX + 7000, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    o = torch.empty((Z * N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    req_to_token_indexs = torch.zeros((10, Z * N_CTX + 7000), dtype=torch.int32, device="cuda")
    max_input_len = N_CTX
    Z = 1
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_req_idx = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_prompt_cache_len = torch.zeros(1, dtype=torch.int32, device="cuda")
    b_prompt_cache_len[0] = 10
    prompt_cache_len = 10

    b_seq_len[0] = 500
    b_req_idx[0] = 0
    req_to_token_indexs[0][: prompt_cache_len + N_CTX] = torch.tensor(
        np.arange(prompt_cache_len + N_CTX), dtype=torch.int32
    ).cuda()

    torch_out = []
    start = 0
    for i in range(Z):
        end = start + b_seq_len[i]
        torch_o = torch_att(
            q[start:end],
            k[start: end + prompt_cache_len],
            v[start: end + prompt_cache_len],
            1,
            b_seq_len[i],
            H,
            D_HEAD,
            prompt_cache_len,
        )
        start = end
        torch_out.append(torch_o)

    torch_out = torch.cat(torch_out, dim=0)
    import time

    torch.cuda.synchronize()
    a = time.time()
    for i in range(10000):
        context_attention_fwd(
            q, k, v, o, b_req_idx, b_start_loc, b_seq_len, b_prompt_cache_len, max_input_len, req_to_token_indexs
        )
    torch.cuda.synchronize()
    b = time.time()
    # print(o.shape, torch_out.shape)
    print((b - a) / 10000)

    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)
