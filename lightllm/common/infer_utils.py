import torch


def init_req_to_token_indexes(
        req_to_token_indexs: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_ready_cache_len: torch.Tensor,
        max_len_in_batch: int,
        alloc_mem_index: torch.Tensor):
    """初始化 req_to_token_indexs 这个 2D-array, 其第一维是 req_idx, 第二维是这个 req 目前在 memory manager 里
       占据的 token index

    Args:
        req_to_token_indexs (torch.Tensor): 要初始化的 req_to_token_indexs tensor
        b_req_idx (torch.Tensor): 一个 1d-tensor, 包含一组 req index
        b_seq_len (torch.Tensor): 一个 1d-tensor, 表示每个 req 需要的 seq len
        b_ready_cache_len (torch.Tensor): 一个 1d-tensor, 表示每个 req 目前已完成分配的 token 数量
        max_len_in_batch (int): 暂时没用到
        alloc_mem_index (torch.Tensor): 一个 1d-tensor, 表示申请下来的所有 memory token index, 需要在本函数中分配到每个 req 里
    """
    start_index = 0
    b_seq_len_numpy = b_seq_len.cpu().numpy()
    b_ready_cache_len_numpy = b_ready_cache_len.cpu().numpy()
    b_req_idx_numpy = b_req_idx.cpu().numpy()
    for i in range(len(b_seq_len)):
        cur_seq_len = b_seq_len_numpy[i]
        cur_ready_cache_len = b_ready_cache_len_numpy[i]
        req_to_token_indexs[b_req_idx_numpy[i], cur_ready_cache_len:cur_seq_len] = alloc_mem_index[
            start_index: start_index + cur_seq_len - cur_ready_cache_len
        ]
        start_index += cur_seq_len - cur_ready_cache_len
    return
