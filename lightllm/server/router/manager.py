from lightllm.server.metrics import monitor
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.server.router.token_load import TokenLoad
from lightllm.utils.log_utils import init_logger
from ..tokenizer import get_tokenizer
from .pause_strategy import Fcfs, select_paused_reqs
from .stats import Stats
from ..io_struct import BatchTokenIdOut, AbortReq, ReqRunStatus, FinishStatus, ReqDetokenizationState
from .dynamic_prompt.radix_cache import RadixCacheReadOnlyClient
from .dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.infer_utils import calculate_time
from rpyc.utils.classic import obtain
from .req_queue import build_req_queue
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from ..multimodal_params import MultimodalParams
from ..io_struct import Req, NormalReq, SplitFuseReq, Batch
from ..sampling_params import SamplingParams
from typing import Dict, List, Optional
import zmq.asyncio
import zmq
import copy
import time
import uuid
import uvloop
import asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

monitor.init_router_monitor()
logger = init_logger(__name__)


class RouterManager:
    def __init__(self, args, router_port, detokenization_port, model_rpc_ports):
        self.args = args
        self.model_weightdir = args.model_dir
        self.world_size = args.tp
        self.load_way = args.load_way
        self.mode = args.mode
        self.max_total_token_num = args.max_total_token_num
        # 用共享内存进行共享，router 模块读取进行精确的调度估计
        self.shared_can_use_token_num = SharedInt(f"{args.nccl_port}_mem_manger_can_use_token_num")
        # 初始化 radix_cache_client 用于读取 prompt cache 的管理信息
        self.radix_cache_client = None
        if self.args.use_dynamic_prompt_cache:
            self.radix_cache_client = RadixCacheReadOnlyClient(str(args.nccl_port), self.max_total_token_num, tp_id=0)

        # 共享变量，用于存储router端调度分析得到的机器负载信息
        self.shared_token_load = TokenLoad(f"{str(args.nccl_port)}_shared_token_load")
        self.shared_token_load.set_current_load(0.0)
        self.shared_token_load.set_logical_max_load(0.0)
        self.shared_token_load.set_dynamic_max_load(0.0)

        self.pause_strategy = Fcfs()
        self.running_batch: Batch = None
        self.eos_id = args.eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = args.router_max_wait_tokens

        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")

        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.is_splitfuse_mode = args.splitfuse_mode
        self.splitfuse_block_size = args.splitfuse_block_size

        self.stats_tool = Stats(not args.disable_log_stats, args.log_stats_interval)
        return

    async def wait_to_model_ready(self):
        # 初始化模型
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            kvargs = {
                "rank_id": rank_id,
                "world_size": self.world_size,
                "weight_dir": self.model_weightdir,
                "load_way": self.load_way,
                "max_total_token_num": self.max_total_token_num,
                "mode": self.mode,
                "max_req_num": self.args.running_max_req_size + 8,
                "max_seq_length": self.args.max_req_total_len + 8,  # 留一点余量
                "nccl_port": self.args.nccl_port,
                "is_splitfuse_mode": self.is_splitfuse_mode,
                "splitfuse_block_size": self.splitfuse_block_size,
                "is_token_healing": self.args.token_healing_mode,
                "return_all_prompt_logprobs": self.args.return_all_prompt_logprobs,
                "use_dynamic_prompt_cache": self.args.use_dynamic_prompt_cache,
                "data_type": self.args.data_type,
                "eos_id": self.eos_id,
                "beam_mode": self.args.beam_mode,
                "diverse_mode": self.args.diverse_mode,
            }
            logger.info("init model rank %s, kvargs = %s", rank_id, kvargs)
            init_model_ret.append(self.model_rpcs[rank_id].init_model(kvargs))

        await asyncio.gather(*init_model_ret)

        self.req_queue = build_req_queue(self.args, self)
        logger.info(f"use req queue {self.req_queue.__class__.__name__}")
        return

    def add_req(
        self,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        group_req_id: int,
    ):
        logger.info("[router managar] addreq 开始执行")
        req_group = []
        for i in range(sampling_params.best_of):
            if self.is_splitfuse_mode:
                logger.info("[router managar][add_req] best_of <%s> 以 splitfuse 模式开始执行, 因此请求被打成 SplitFuseReq", i)
                req = SplitFuseReq(
                    group_req_id + i,
                    copy.deepcopy(prompt_ids),
                    sampling_params,
                    multimodal_params,
                    self.splitfuse_block_size,
                )
            else:
                logger.info("[router managar][add_req] best_of <%s> 以非 SplitFuse 模式开始执行, 因此请求被打成 NormalReq", i)
                req = NormalReq(group_req_id + i, copy.deepcopy(prompt_ids), sampling_params, multimodal_params)
            req_group.append(req)
        logger.info("[router manager][add_req] req_group is append to req_queue")
        self.req_queue.extend(req_group)
        self.send_to_detokenization.send_pyobj(
            ReqDetokenizationState(
                group_req_id,
                prompt_ids,
                sampling_params.max_new_tokens,
                sampling_params.ignore_eos,
                sampling_params.skip_special_tokens,
                sampling_params.add_spaces_between_special_tokens,
                sampling_params.print_eos_token,
                sampling_params.best_of,
            )
        )
        return

    async def abort(self, group_req_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if convert_sub_id_to_group_id(req.request_id) == group_req_id:
                    req.finish_status = FinishStatus.FINISHED_ABORT
        for req in self.req_queue.waiting_req_list:
            if convert_sub_id_to_group_id(req.request_id) == group_req_id:
                req.finish_status = FinishStatus.FINISHED_ABORT
        return

    async def loop_for_fwd(
        self,
    ):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    token_ratio1 = self.get_used_tokens() / self.max_total_token_num
                    token_ratio2 = (
                        self.max_total_token_num - self.shared_can_use_token_num.get_value()
                    ) / self.max_total_token_num
                    logger.debug(
                        f"current batch size: {len(self.running_batch.reqs)} \n"
                        f"paused req num: {len(self.req_queue.pause_req_dict)} \n"
                        f"token used ratio: {token_ratio1} not contain prompt cache tree unrefed tokens\n"
                        f"token used ratio: {token_ratio2} contain prompt cache tree unrefed tokens"
                    )
                    self.shared_token_load.set_current_load(token_ratio1)
                    self.req_queue.update_token_load(self.running_batch)
                    pass
                self.stats_tool.print_stats()
                monitor.gauge_set("lightllm_batch_current_size", len(self.running_batch.reqs))
                monitor.gauge_set("lightllm_batch_pause_size", len(self.req_queue.pause_req_dict))
                monitor.gauge_set("lightllm_queue_size", len(self.req_queue.waiting_req_list))
            else:
                self.shared_token_load.set_dynamic_max_load(0.0)
                self.shared_token_load.set_current_load(0.0)
                monitor.gauge_set("lightllm_batch_current_size", 0.0)
                monitor.gauge_set("lightllm_batch_pause_size", 0.0)
                monitor.gauge_set("lightllm_queue_size", 0.0)

            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def _step(self):
        """
        事件处理循环
        """
        # logger.info("_step ################################################################")
        # 删除所有已经 finished 的 req
        # 当前无运行请求时
        if self.running_batch is None:
            # logger.info("[_step] 当前无运行请求时, generate new batch")
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                logger.info("[_step] 有newbatch产生, 更新 stat_pool(因为新增了一些 prompt tokens)")
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                logger.info("[_step] 有newbatch产生, 执行 _prefill_batch")
                await self._prefill_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        # 有运行请求，但是已经到了可以调度新的请求合并推理的时机
        if self.has_wait_tokens >= self.max_wait_tokens:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
            self.has_wait_tokens = 0
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)
                await self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                return

        # 正常 decode 阶段， 如果可以直接decode就直接decode，否则通过暂停策略暂停一些请求
        # 释放一些管理的 token
        # 确认 batch 内一次 decode 需要额外占用的 token 数 + 当前已使用的 token 数是否会超标
        if self._can_decode(self.running_batch):
            self.stats_tool.count_output_tokens(self.running_batch)  # 如果不会超标, 则把 batch 这次会新增的 out token 刷新到 stats pool
            await self._decode_batch(self.running_batch)  # 执行 decode_batch
            self._filter_runing_batch()  # 从 running_batch 中过滤掉 finish 的 req
            self.has_wait_tokens += 1  # 计数 + 1, 计数到阈值才会做 prefill
            return
        else:  # 无法直接 decode, 需要 pause 一些历史请求, 抢占对应的 token
            # pause strategy
            paused_reqs = select_paused_reqs(  # 选取需要暂停的 req, 注意这个函数内顺便已经帮你把 req 标记成暂停状态了, (感觉不是很合理..
                self.running_batch, self.pause_strategy, self.req_queue, self.max_total_token_num
            )
            await self._pause_reqs(self.running_batch, paused_reqs)
            logger.debug(f"pasued req num: {len(self.req_queue.pause_req_dict)}")
            self.has_wait_tokens = 0
            return
        return

    async def _init_batch(self, batch: Batch):
        logger.debug("[_init_batch] begin, 初始化 batch 状态, 做的事情如下")
        logger.debug("[_init_batch] 初始化 batch 状态, 构建 batch 内 reqs 的 rpc obj, 用来给 model 推理的 rpc 进程发送")
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        # logger.debug("[_init_batch], 准备开始观察 init_batch 的内部逻辑,TODO##################################")
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_req_status = obtain(ans[0])
        else:
            req_to_req_status = ans[0]
        # 请求状态， 当前占用的kv的长度， 当前输出token的数量， 输出的token的id和元信息列表， 是否推理结束的状态， 额外保留参数
        self._update_init_status_to_batch(batch, req_to_req_status)
        logger.debug(f"Init Batch: {batch.simple_log()} \n")
        return

    async def _prefill_batch(self, batch: Batch):
        logger.debug("[_prefill_batch], begin")
        await self._init_batch(batch)
        if not self.is_splitfuse_mode:
            # 在 非 splitfuse 模式下，才需要真的执行 prefill 的操作。
            rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
            ans = await asyncio.gather(*rets)
            # 此处任取一个对象结果即可, 毕竟 model 是 tp 的, 即结果都是 all_reduce 后再发出来的
            if self.world_size != 1:
                req_to_out_status = obtain(ans[0])
            else:
                req_to_out_status = ans[0]
            # 利用返上来的 res_to_out_status 更新 router 进程内的 req 的状态
            self._update_out_status_to_batch(batch, req_to_out_status)
            # 获得 unfinished_req_id 以及 finished_req_ids 两个 list
            unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status()
            #
            self._send_to_detokenization_proc(batch, req_to_out_status)
            batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
            await self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
        return

    async def _decode_batch(self, batch: Batch):
        logger.debug("[_decode_batch], begin")
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_status = obtain(ans[0])
        else:
            req_to_out_status = ans[0]

        self._update_out_status_to_batch(batch, req_to_out_status)
        unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status()
        self._send_to_detokenization_proc(batch, req_to_out_status)
        # 当前进程在发送结果后，还需要调用 batch.filter_out_finished_req 将batch 内 finish 的 req 去除，
        # 这些 req 不应该参与后续的 decode 动作（前面提到的, 处于 abort 状态的 pause req，也会在恢复并完成 prefill 后，在此刻结束）
        batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
        # 利用 unfinished req ids 和 finished req ids 更新 batch 状态
        await self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
        return

    async def _filter_batch(self, batch: Batch, unfinished_req_ids, finished_req_ids: List):
        rets = [
            self.model_rpcs[tp_rank].filter_batch(batch.batch_id, unfinished_req_ids, finished_req_ids)
            for tp_rank in range(self.world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [
            self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _pause_reqs(self, batch: Batch, pasue_reqs):
        pasue_reqs_info = [(r.request_id, r.req_status) for r in pasue_reqs]
        rets = [
            self.model_rpcs[tp_rank].pause_reqs(batch.batch_id, pasue_reqs_info) for tp_rank in range(self.world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, unfinished_req_ids, finished_req_ids):
        """向底层 model_rpc 进程同步 router 主进程更新后的 batch 状态, 保持上下 batch 的状态同步"""
        if len(finished_req_ids) != 0:
            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch, unfinished_req_ids, finished_req_ids)
        return

    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return

    def _update_init_status_to_batch(self, batch: Batch, req_to_req_status):
        self._update_out_status_to_batch(batch, req_to_req_status)
        return

    def _update_out_status_to_batch(self, batch: Batch, req_to_out_status):
        """利用返上来的 res_to_out_status 更新 batch 内 req 的状态"""
        new_batch_decode_need_tokens = 0  # 只有在 splitfuse 模式下有意义
        for req_id, (
            req_status,
            cur_kv_len,
            cur_output_len,
            token_info_list,
            finish_status_value,
            extral_info,
        ) in req_to_out_status.items():
            logger.debug("req_id = %s, req_to_req_status = %s(请求状态), %s(当前占用的kv的长度), %s(当前输出token的数量)"
                         ", %s(输出的token的id和元信息列表), %s(是否推理结束的状态)"
                         ", %s(额外保留参数)", req_id, req_status, cur_kv_len, cur_output_len, token_info_list, finish_status_value, extral_info)
            req: Req = batch.id_to_reqs[req_id]
            req.req_status = req_status
            req.cur_kv_len = cur_kv_len
            req.cur_output_len = cur_output_len
            # 暂时不维护 output_ids 和 output_metadata_list
            # for (new_token_id, new_gen_metadata) in token_info_list:
            #     req.output_ids.append(new_token_id)
            #     req.output_metadata_list.append(new_gen_metadata)
            # 当没有被 aborted 的时候，才更新请求状态。
            if not req.finish_status.is_aborted():
                req.finish_status = FinishStatus(finish_status_value)
            new_batch_decode_need_tokens += req.get_decode_need_tokens()

        batch.batch_decode_need_tokens = new_batch_decode_need_tokens
        return

    def _can_decode(self, batch: Batch):
        return batch.batch_decode_need_tokens + self.get_used_tokens() <= self.max_total_token_num

    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        """将 req_ans 结果推送到 detokenization 进程, 完成剩余工作;
        batch 传进来是因为 req_ans 内部只包含了 req_id, 需要从 batch 内捞取 req 的 finish status"""
        batch_out = BatchTokenIdOut()
        for req_id, (_, _, _, token_info_list, _, _) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            for idx, (new_token_id, new_gen_metadata) in enumerate(token_info_list):
                # req.finish_status 传输 value值 不传送对象，可以减少序列化对象的大小。
                if idx == len(token_info_list) - 1:
                    batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.finish_status.value))
                else:
                    batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, FinishStatus.NO_FINISH))

        self.send_to_detokenization.send_pyobj(batch_out)
        return

    def get_used_tokens(self):
        if self.args.use_dynamic_prompt_cache:
            return (
                self.max_total_token_num
                - self.shared_can_use_token_num.get_value()
                - self.radix_cache_client.get_unrefed_tokens_num()
            )
        else:
            return self.max_total_token_num - self.shared_can_use_token_num.get_value()

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 4:
                prompt_ids, sampling_params, multimodal_params, group_req_id = recv_req
                logger.info("[router manager] 收到一个 http 请求, 准备开始执行, "
                            "请求[prompt_ids, sampling_params, multimodal_params, group_req_id]为 %s", recv_req)
                self.add_req(prompt_ids, sampling_params, multimodal_params, group_req_id)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                group_req_id = abort_req.group_req_id
                await self.abort(group_req_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_router_process(args, router_port, detokenization_port, model_rpc_ports, pipe_writer):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    try:
        router = RouterManager(
            args, router_port=router_port, detokenization_port=detokenization_port, model_rpc_ports=model_rpc_ports
        )

        asyncio.run(router.wait_to_model_ready())
    except BaseException:
        import traceback
        import sys

        etype, evalue, tb = sys.exc_info()
        err_str = "\n".join(traceback.format_exception(etype, evalue, tb))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send("init ok")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
