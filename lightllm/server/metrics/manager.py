import asyncio
import rpyc
import time
import threading
from typing import Union
from rpyc.utils.classic import obtain
from .metrics import Monitor
from prometheus_client import generate_latest
import multiprocessing.shared_memory as shm
from concurrent.futures import ThreadPoolExecutor
import functools
from rpyc import async_, SocketStream
import queue
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def connect_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.update({"timeout": 30})  # update the default timeout (3) to 30s
        return func(*args, **kwargs)

    return wrapper


SocketStream._connect = connect_decorator(SocketStream._connect)


class MetricServer(rpyc.Service):
    def __init__(self, args) -> None:
        super().__init__()
        self.monitor = Monitor(args)
        self.interval = args.push_interval

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_counter_inc(self, name: str, label: str = None) -> None:
        return self.monitor.counter_inc(name, label)

    def exposed_histogram_observe(self, name: str, value: float, label: str = None) -> None:
        return self.monitor.histogram_observe(name, value, label)

    def exposed_gauge_set(self, name: str, value: float) -> None:
        return self.monitor.gauge_set(name, value)

    def exposed_generate_latest(self) -> bytes:
        data = generate_latest(self.monitor.registry)
        return data

    def push_metrics(self):
        while True:
            self.monitor.push_metrices()
            time.sleep(self.interval)


class MetricClient(threading.Thread):
    def __init__(self, port):
        super().__init__()
        self.port = port
        self.conn = rpyc.connect("localhost", self.port)

        def async_wrap(f):
            f = rpyc.async_(f)

            async def _func(*args, **kwargs):
                ans = f(*args, **kwargs)
                await asyncio.to_thread(ans.wait)
                return ans.value

            return _func

        self._generate_latest = async_wrap(self.conn.root.generate_latest)

        self.task_queue = queue.Queue(maxsize=1024)
        self.daemon = True
        self.start()

    async def generate_latest(self):
        ans = await self._generate_latest()
        return ans

    def counter_inc(self, *args, **kwargs):
        def inner_func():
            return self.conn.root.counter_inc(*args, **kwargs)

        self._append_task(inner_func)
        return

    def histogram_observe(self, *args, **kwargs):
        def inner_func():
            return self.conn.root.histogram_observe(*args, **kwargs)

        self._append_task(inner_func)
        return

    def gauge_set(self, *args, **kwargs):
        def inner_func():
            return self.conn.root.gauge_set(*args, **kwargs)

        self._append_task(inner_func)
        return

    def _append_task(self, task_func):
        try:
            self.task_queue.put_nowait(task_func)
        except queue.Full as e:
            logger.warning(f"monitor task queue is full, error {str(e)}")
        return

    def run(self):
        while True:
            task_func = self.task_queue.get()
            try:
                task_func()
            except Exception as e:
                logger.error(f"monitor error {str(e)}")


def start_metric_manager(port: int, args, pipe_writer):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    service = MetricServer(args)
    if args.metric_gateway is not None:
        push_thread = threading.Thread(target=service.push_metrics)
        push_thread.start()  # 启动推送任务线程

    from rpyc.utils.server import ThreadedServer

    t = ThreadedServer(service, port=port)
    pipe_writer.send("init ok")
    t.start()
