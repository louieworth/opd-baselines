"""Give each colocated vLLM engine its own port range below the kernel's ephemeral ports.

vLLM picks its torch.distributed port by binding port 0, closing it, and binding again
later. With many engines starting at once on a node, another process can take that
ephemeral port in between (EADDRINUSE). With VLLM_PORT set, vLLM searches upward from
it instead, so disjoint per-engine ranges below 32768 cannot collide.
"""

import os
import re

BASE_PORT = 20000
TEACHER_OFFSET = 5000
PORTS_PER_ENGINE = 50
# verl names servers <prefix>server[_teacher]_<replica_rank>_<node_rank>.
SERVER_NAME = re.compile(r'server_(teacher_)?(\d+)_\d+$')


def engine_port(actor_name):
    match = SERVER_NAME.search(actor_name or '')
    if not match:
        return None
    return BASE_PORT + (TEACHER_OFFSET if match.group(1) else 0) + PORTS_PER_ENGINE * int(match.group(2))


def install():
    import ray
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
    launch = vLLMHttpServer.launch_server
    if getattr(launch, 'pins_vllm_port', False):
        return

    async def launch_server(self, *args, **kwargs):
        port = engine_port(ray.get_runtime_context().get_actor_name())
        if port is not None and 'VLLM_PORT' not in os.environ:
            os.environ['VLLM_PORT'] = str(port)  # inherited by the engine core and its workers
        return await launch(self, *args, **kwargs)

    launch_server.pins_vllm_port = True
    vLLMHttpServer.launch_server = launch_server
