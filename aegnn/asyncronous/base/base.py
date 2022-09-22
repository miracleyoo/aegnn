from contextlib import contextmanager
import logging
import time


def add_async_graph(module, r: float = None, log_flops: bool = False, log_runtime: bool = False):
    """ 只是单纯添加了一个Asyn的radius，初始化一下Asyn相关变量到空。
    """
    module.asy_graph = None
    module.asy_flops_log = [] if log_flops else None
    module.asy_runtime_log = [] if log_runtime else None
    if r is not None:
        module.asy_radius = r
    return module


def make_asynchronous(module, initialization_func, processing_func):
    """ 将module的forward改为Asyn的forward。这个Asyn的forward即如async_context所述，
        如果还没有Asyn图，创建之；反之，执行之
    """
    def async_forward(*args, **kwargs):
        with async_context(module, initialization_func, processing_func) as func:
            output = func(module, *args, **kwargs)
        return output

    module.forward = async_forward
    return module


@contextmanager
def async_context(module, initialization_func, processing_func):
    """ 创造Async环境，如果该module中没有Asyn图，创建之；反之，执行之
    """
    do_log_runtime = getattr(module, "asy_runtime_log", None) is not None
    start_time = time.time() if do_log_runtime else None

    if module.asy_graph is None:
        logging.debug(f"Graph initialization of module {module}")
        yield initialization_func
    else:
        logging.debug(f"Calling processing of module {module}")
        yield processing_func

    if do_log_runtime:
        dt = time.time() - start_time
        module.asy_runtime_log.append(dt)
