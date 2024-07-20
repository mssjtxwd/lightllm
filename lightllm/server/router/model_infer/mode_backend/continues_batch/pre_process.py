import torch
import numpy as np
from lightllm.server.router.model_infer.infer_batch import requests_mapping, InferReq, InferBatch
from lightllm.server.io_struct import ReqRunStatus
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.common.mem_manager import MemoryManager
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# @calculate_time(show=True, min_cost_ms=1)


def prepare_prefill_inputs(batch: InferBatch, radix_cache: RadixCache, is_multimodal=False):
    """获得调用 model prefill 所需要的所有参数"""
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    for request_id in batch.request_ids:
        req: InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING

        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)  # 收集每个 req 的多模态参数
        nopad_b_req_idx.append(req.req_idx)  # 把 req 的 req_idx 连成一个 1d-tensor
        # 因为会把 req 的 input_token_ids 连成一个 1d-tensor(即 nopad 形式), 所以为了分离他们,
        # 需要记录每个 req 对应的 input_token_ids 在 input_ids 里的 start_loc
        nopad_b_start_loc.append(start_loc)

        # 每个 req 的 input_token_ids 长度即 req 现状的 seq_len
        seq_len = len(req.input_token_ids)
        # input_token_len 表示这一次要生成的 token_len, 在 prefill 的大多数情况下, cur_kv_len 应该都为 0, 但如果使用了
        # prompt_cache, 则对于一个 pause req, 在前面的 init_batch 阶段, 会把这个 req 内部的 kvcache 恢复回来, 此时的
        # req.cur_kv_len 就会被更新成 pause 之前的状态
        # 此处的处理主要也是针对这个问题
        input_token_len = seq_len - req.cur_kv_len

        # 同样是特判 prompt cache 下的情况, 真正的 input_id 也是去掉了已经入 kv_cache 外的部分
        input_id = req.input_token_ids[req.cur_kv_len:]

        nopad_b_seq_len.append(seq_len)  # 简单的把 seq_len 搞成一个1d-tensor
        input_ids.append(input_id)  # 拼接 input_id, 后续 concat 后就可以做成 nopad 的 1d-tensor
        nopad_total_token_num += seq_len  # 统计 req 的 seq_len 之和
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_token_len)  # 统计 input_token_len(这一次要生成的 token_len) 的最大值
        b_ready_cache_len.append(req.cur_kv_len)  # 统计每个 req 中 ready 的, 在 cache 内的 token 的数量
        start_loc += input_token_len  # 更新 start_loc

    input_ids = np.concatenate(input_ids, dtype=np.int64)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cuda")
    kwargs = {
        "batch_size": len(batch),  # batchsize
        "total_token_num": nopad_total_token_num,  # batch 内总计的 token 数量
        # batch 内这一波要推理的 token 数的最大值(prefill 阶段其实就是 max([len(req.input_token_ids) for req in batch.reqs)])
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,  # 拼接好的 1d-tensor, 可根据 start_loc 提取出每个 req 的 input_ids
        "b_req_idx": nopad_b_req_idx,  # 拼接好的1d-tensor, 每个元素表示一个 req 的 req_idx
        "b_start_loc": nopad_b_start_loc,  # 拼接好的1d-tensor, 每个元素表示一个 req 的 input_ids 在 全集的 input_ids 中的 start 位置
        "b_seq_len": nopad_b_seq_len,  # 拼接好的 1d-tensor, 每个元素表示一个 req 的 seq_len
        "b_ready_cache_len": b_ready_cache_len,  # 拼接好的 1d-tensor, 每个元素表示一个 req 中已经在 kv cache 中的 token 数量
        "is_prefill": True,
    }
    if is_multimodal:
        kwargs["multimodal_params"] = batch_multimodal_params

    # dynamic prompt cache 准备 token
    if radix_cache is not None:
        radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])

    return kwargs, run_reqs


# @calculate_time(show=True, min_cost_ms=1)
def prepare_decode_inputs(batch: InferBatch, radix_cache: RadixCache):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    for request_id in batch.request_ids:
        req: InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING
        run_reqs.append(req)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        input_id = req.input_token_ids[-1]
        seq_len = len(req.input_token_ids)
        assert req.cur_kv_len == seq_len - 1
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    kwargs = {
        "batch_size": len(batch),
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "is_prefill": False,
    }
    # dynamic prompt cache 准备 token
    if radix_cache is not None:
        radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])

    return kwargs, run_reqs
