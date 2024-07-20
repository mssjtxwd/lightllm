import time
import uuid
import numpy as np
from typing import List
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.io_struct import Batch, Req
from lightllm.server.io_struct import ReqRunStatus
from lightllm.server.router.req_queue.base_queue import BaseQueue
from lightllm.utils.log_utils import init_logger
logger = init_logger(__name__)


class ContinuesBatchQueue(BaseQueue):
    def __init__(self, args, router) -> None:
        super().__init__(args, router)

    def _init_cache_list(self, current_batch: Batch, is_busy):
        if current_batch is not None:
            self.cache_len_list = [
                req.get_tuple_tokens(is_busy, self.router_max_new_token_len) for req in current_batch.reqs
            ]
        else:
            self.cache_len_list = []
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req: Req, is_busy, new_batch_first_router_need_tokens):
        """用来判定当前是否还能 add 一个新 req """
        self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len))  # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        # 判定标准1: 加入新 req 后, 峰值 token 数不会超过阈值
        # 峰值 token 数推算方式
        # 举例, 目前有 3个 req a, b, c, 已推数量 8, 9, 10, 未推数量 12, 11, 10
        # 峰值 token 数只会发生在任意 req 达到上限时
        # 如果 c 到达上限, 此时 token 总量为 8 + 9 + 10 + 10 * 3
        # 如果 b 到达上限, 此时 c 已经结束, 只需考察 a,b, 总量为 8 + 9 + 11 * 2
        # 同理, 如果c 到达上限, 总量为 8 + 12
        # 以上即为此处实现
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        ok_token_num = need_max_token_num < self.max_total_tokens

        # 判定标准2: 运行状态(包括 pause) 的 req 总数小于阈值
        if req.req_status != ReqRunStatus.PAUSED_AND_OFFLOAD:
            ok_req_num = len(self.cache_len_list) + len(self.pause_req_dict) <= self.running_max_req_size
        else:
            # 因为存在重复的项
            ok_req_num = len(self.cache_len_list) + len(self.pause_req_dict) - 1 <= self.running_max_req_size

        new_batch_first_router_need_tokens += req.get_first_router_need_tokens()
        # prefill ok
        # 判定标准3: 加上新 req 的 prefill 阶段所需 token 数后[注意, 对于 pause 的 req, 其 prefill 阶段的 token 数是 input_len + has_out_len], prefill 阶段的所需 token 数仍然小于阈值
        ok_prefill = new_batch_first_router_need_tokens <= self.batch_max_tokens

        if ok_token_num and ok_req_num and ok_prefill:
            self.router.shared_token_load.set_dynamic_max_load(need_max_token_num / self.max_total_tokens)
            return True, new_batch_first_router_need_tokens
        else:
            return False, new_batch_first_router_need_tokens

    # @calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch: Batch):
        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        # logger.info("ContinuesBatchQueue.generate_new_batch, 开始阶段")
        exist_req_num = (0 if current_batch is None else len(current_batch.reqs)) + len(self.pause_req_dict)
        # logger.info("exist_req_num = %s, 如果 >= running_max_req_size( = %s) 则认为 is full", exist_req_num, self.running_max_req_size)
        req_is_full = exist_req_num >= self.running_max_req_size
        if req_is_full:
            return None

        # 判断服务是否繁忙
        is_busy = self.is_busy()

        self._init_cache_list(current_batch, is_busy)
        can_run_list = []
        new_batch_first_router_need_tokens = 0  # 主要是对 prefill 大块计算时候的token数量限制
        aborted_count = 0
        for req in self.waiting_req_list:
            if req.finish_status.is_aborted() and req.req_status == ReqRunStatus.WAIT_IN_QUEUE:
                # 由于管理的复杂性，只有没有被调度运行过的请求可以因为abort直接在队列中忽略掉.
                # 暂停的请求需要恢复后，由 router manager 部分来过滤。暂时保持这种处理方法, 否则会导致管理token和管理req对象的泄漏
                aborted_count += 1
                continue
            ok_insert, new_batch_first_router_need_tokens = self._can_add_new_req(
                req, is_busy, new_batch_first_router_need_tokens
            )
            if ok_insert:
                can_run_list.append(req)
                if req.req_status == ReqRunStatus.PAUSED_AND_OFFLOAD:
                    self.pause_req_dict.pop(req.request_id)
            else:
                break
        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            return new_batch
        else:
            return None

    def calcu_batch_token_load(self, current_batch: Batch):
        if current_batch is None:
            return 0.0
        is_busy = self.is_busy()
        self._init_cache_list(current_batch, is_busy)
        self.cache_len_list.sort(key=lambda x: -x[1])
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        return need_max_token_num / self.max_total_tokens
