from lightllm.utils.log_utils import init_logger
from ..metrics import monitor
from ..sampling_params import SamplingParams
from ..req_id_generator import convert_sub_id_to_group_id
from ..embed_cache.utils import get_shm_name_data, create_shm
from ..io_struct import BatchStrOut, AbortReq, FinishStatus
from ..tokenizer import get_tokenizer
import sys
import zmq
import zmq.asyncio
import asyncio
import uvloop
import rpyc
import time
import hashlib

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = init_logger(__name__)


class HttpServerManager:
    def __init__(
        self,
        args,
        router_port,
        cache_port,
        httpserver_port,
        visual_port,
        enable_multimodal,
    ):
        self.args = args
        context = zmq.asyncio.Context(2)
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"tcp://127.0.0.1:{router_port}")

        self.enable_multimodal = enable_multimodal
        if self.enable_multimodal:
            self.cache_client = rpyc.connect("localhost", cache_port)
            self.send_to_visual = context.socket(zmq.PUSH)
            self.send_to_visual.connect(f"tcp://127.0.0.1:{visual_port}")

        self.recv_from_detokenization = context.socket(zmq.PULL)
        self.recv_from_detokenization.bind(f"tcp://127.0.0.1:{httpserver_port}")

        self.tokenizer = get_tokenizer(args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code)

        self.req_id_to_out_inf = {}  # value type (out_str, metadata, finished, event)

        self.total_token_num = args.max_total_token_num
        self.max_req_input_len = args.max_req_input_len
        self.max_req_total_len = args.max_req_total_len
        monitor.init_httpserver_monitor(args)

        return

    # connect cache server, calculate md5, alloc resource, return uuid
    async def _alloc_resource(self, data, num):
        md5sum = hashlib.md5(data).hexdigest()
        wait_time = 1
        while True:
            record = self.cache_client.root.alloc(md5sum, num)
            # hit or new
            if record:
                uid = record["id"]
                if not self.cache_client.root.get_item_data(uid):
                    create_shm(get_shm_name_data(uid), data)
                    self.cache_client.root.set_item_data(uid)
                return record
            # cache full
            else:
                await asyncio.sleep(wait_time)
                wait_time = min(wait_time + 2, 9)

    async def _alloc_multimodal_resources(self, multimodal_params):
        for img in multimodal_params.images:
            record = await self._alloc_resource(img.read(), self.tokenizer.get_image_token_length(img))
            img.uuid = record["id"]
            img.token_id = record["token_id"]
            img.token_num = record["token_num"]

    async def _release_multimodal_resources(self, multimodal_params):
        if multimodal_params is not None:
            for img in multimodal_params.images:
                if img.uuid is not None:
                    self.cache_client.root.release(img.uuid)

    def tokens(self, prompt):
        prompt_ids = self.tokenizer.encode(prompt)
        return len(prompt_ids)

    async def generate(
        self, prompt, sampling_params: SamplingParams, group_request_id, multimodal_params, request=None
    ):
        # 统计信息变量
        start_time = time.time()
        out_token_counter = 0
        first_token_cost_ms = sys.float_info.max
        is_first_token = True

        if self.enable_multimodal:
            assert len(multimodal_params.images) <= self.args.cache_capacity, "too many images!"
            await self._alloc_multimodal_resources(multimodal_params)
            prompt_ids = self.tokenizer.encode(prompt, multimodal_params)
        else:
            logger.info("[httpserver.manager] http线程负责 tokenizer, 这里是 tokenizer 的部分")
            prompt_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_ids)
        monitor.histogram_observe("lightllm_request_input_length", prompt_tokens)
        monitor.histogram_observe("lightllm_request_max_new_tokens", sampling_params.max_new_tokens)
        if prompt_tokens > self.max_req_input_len:
            # use long_truncation_mode to truncate long input len req.
            logger.info("[httpserver.manager] 超过了设置的 req 的最大输入大小")
            if self.args.long_truncation_mode is None:
                logger.info("[httpserver.manager] 不允许截断, 则报错")
                raise ValueError(f"the input prompt token len {prompt_tokens} is too long > {self.max_req_input_len}")
            elif self.args.long_truncation_mode == "head":
                logger.info("[httpserver.manager] 允许截断, 那走截断, 策略是 head")
                prompt_ids = prompt_ids[-self.max_req_input_len:]
                prompt_tokens = len(prompt_ids)
            elif self.args.long_truncation_mode == "center":
                logger.info("[httpserver.manager] 允许截断, 那走截断, 策略是 center")
                prompt_ids = (
                    prompt_ids[0: self.max_req_input_len // 2]
                    + prompt_ids[-(self.max_req_input_len - self.max_req_input_len // 2):]
                )
                prompt_tokens = len(prompt_ids)
                assert prompt_tokens == self.max_req_input_len
            else:
                assert False, "error args"

        req_total_len = prompt_tokens + sampling_params.max_new_tokens
        if req_total_len > self.max_req_total_len:
            raise ValueError(
                f"the req total len (input len + output len) is too long > max_req_total_len:{self.max_req_total_len}"
            )
        if req_total_len + 1 > self.total_token_num:
            raise ValueError(f"the req token total len + 1 is too long > max_total_token_num:{self.total_token_num}")

        logger.info("[httpserver.manager] 根据 sampling params, 将 stop sentences 转换成 token ids[Q: stop sentences 如何产生作用]")
        sampling_params.stop_sentences_to_token_ids(self.tokenizer)

        req_status = ReqStatus(group_request_id, multimodal_params)
        event = req_status.event
        self.req_id_to_out_inf[group_request_id] = req_status

        if self.enable_multimodal:
            self.send_to_visual.send_pyobj((prompt_ids, sampling_params, multimodal_params, group_request_id))
        else:
            logger.info("[httpserver.manager] 发送 prompt 到 router, prompt_ids len = %s, prompt_ids = %s", len(prompt_ids), prompt_ids)
            self.send_to_router.send_pyobj((prompt_ids, sampling_params, multimodal_params, group_request_id))

        unfinished_count = sampling_params.best_of

        while True:
            try:
                logger.info("[httpserver.manager] 默默等待 detokenization 把结果发回来")
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass

            if request is not None and await request.is_disconnected():
                logger.info("[httpserver.manager] abort 原请求")
                await self.abort(group_request_id)
                raise Exception(f"req_id {group_request_id} disconnected")

            async with req_status.lock:
                event.clear()
                if len(req_status.out_token_info_list) == 0:
                    continue

                for sub_req_id, out_str, metadata, finish_status in req_status.out_token_info_list:
                    metadata["prompt_tokens"] = prompt_tokens
                    out_token_counter += 1
                    first_token_cost_ms = (time.time() - start_time) * 1000 if is_first_token else first_token_cost_ms
                    is_first_token = False

                    logger.info(
                        "[httpserver.manager] 吐出一些 token: out_str = %s, metadata = %s, finish_status = %s",
                        out_str,
                        metadata,
                        finish_status)

                    yield sub_req_id, out_str, metadata, finish_status

                    # 如果有子请求完成，就更新计数
                    if finish_status.is_finished():
                        unfinished_count -= 1

                    # 所有子请求完成后，就删除占用的资源
                    if unfinished_count == 0:
                        logger.info("httpserver.manager] 所有子请求完成后，就删除占用的资源")
                        try:
                            del self.req_id_to_out_inf[group_request_id]
                            await self._release_multimodal_resources(multimodal_params)
                        except BaseException:
                            pass
                        total_cost_time_ms = (time.time() - start_time) * 1000
                        mean_per_token_cost_time_ms = (total_cost_time_ms - first_token_cost_ms) / out_token_counter
                        logger.debug(
                            f"req_id:{group_request_id},start:{start_time}s,first_token_cost:{first_token_cost_ms}ms\n"
                            f"total_cost_time:{total_cost_time_ms}ms,out_token_counter:{out_token_counter}\n"
                            f"mean_per_token_cost_time: {mean_per_token_cost_time_ms}ms\n"
                            f"prompt_token_num:{prompt_tokens}"
                        )
                        monitor.histogram_observe("lightllm_request_inference_duration", total_cost_time_ms)
                        monitor.histogram_observe(
                            "lightllm_request_mean_time_per_token_duration", total_cost_time_ms / out_token_counter
                        )
                        monitor.histogram_observe("lightllm_request_first_token_duration", first_token_cost_ms)
                        monitor.histogram_observe("lightllm_request_generated_tokens", out_token_counter)
                        return
                req_status.out_token_info_list.clear()
        return

    async def abort(self, group_request_id):
        abort_req = AbortReq(group_req_id=group_request_id)
        self.send_to_router.send_pyobj(abort_req)
        if self.enable_multimodal:
            self.send_to_visual.send_pyobj(abort_req)
        try:
            req = self.req_id_to_out_inf[group_request_id]
            await self._release_multimodal_resources(req.multimodal_params)
            del self.req_id_to_out_inf[group_request_id]
        except BaseException:
            pass
        logger.warning(f"aborted group_request_id {group_request_id}")
        return

    async def handle_loop(self):
        while True:
            recv_ans: BatchStrOut = await self.recv_from_detokenization.recv_pyobj()
            assert isinstance(recv_ans, BatchStrOut), f"error recv type {type(recv_ans)}"
            for sub_req_id, text, metadata, finish_status in recv_ans.reqs_infs:
                finish_status = FinishStatus(finish_status)
                group_req_id = convert_sub_id_to_group_id(sub_req_id)
                try:
                    if not finish_status.is_aborted():
                        req_status: ReqStatus = self.req_id_to_out_inf[group_req_id]
                        async with req_status.lock:
                            req_status.out_token_info_list.append((sub_req_id, text, metadata, finish_status))
                            req_status.event.set()
                    else:
                        del self.req_id_to_out_inf[group_req_id]
                except BaseException:
                    pass
        return


class ReqStatus:
    def __init__(self, req_id, multimodal_params) -> None:
        self.req_id = req_id
        self.multimodal_params = multimodal_params
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self.out_token_info_list = []
