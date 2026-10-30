"""Resumable, token-exact offline V2 reference traces from the frozen initial teacher."""

import hashlib
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

# Shared (network) cache storage has high per-file latency; overlap that many file operations.
IO_THREADS = 64


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, separators=(',', ':')) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_identity(config):
    """Fingerprint local weights AND tokenizer/template, without hashing multi-GB files."""
    model = Path(config['teacher_model_path']).resolve()
    if not model.is_dir():
        raise ValueError('ADOPD V2 requires a local frozen teacher model directory')
    files = []
    for pattern in ('*.safetensors', '*.bin', '*config.json', 'tokenizer.json', 'vocab.json', 'merges.txt'):
        for path in sorted(model.glob(pattern)):
            stat = path.stat()
            files.append((path.name, stat.st_size, stat.st_mtime_ns))
    if not any(name.endswith(('.safetensors', '.bin')) for name, *_ in files):
        raise ValueError(f'No teacher weight files in {model}')
    return dict(version=1, model=str(model), files=files,
                teacher_budget=config['adopd_teacher_max_completion_length'],
                seed=config['adopd_cache_seed'], temperature=1.0, top_p=1.0, top_k=-1,
                enable_thinking=True)


class TraceCache:
    def __init__(self, root, identity):
        self.identity = identity
        self.fingerprint = digest(identity)
        self.root = Path(root) / self.fingerprint

    def path(self, prompt_ids):
        key = digest(list(prompt_ids))
        return self.root / key[:2] / (key + '.json')

    def read(self, prompt_ids, halt_token_id):
        path = self.path(prompt_ids)
        if not path.is_file():
            raise FileNotFoundError(f'Missing V2 teacher trace: {path}. Run --prepare-teacher-cache first.')
        entry = json.loads(path.read_text())
        trace = entry.get('trace_ids', [])
        if (entry.get('fingerprint') != self.fingerprint or entry.get('prompt_ids') != list(prompt_ids)
                or not trace or any(type(t) is not int or t < 0 for t in trace)
                or trace[-1] != halt_token_id or halt_token_id in trace[:-1]
                or len(trace) > self.identity['teacher_budget']):
            raise ValueError(f'Invalid or incompatible V2 teacher trace: {path}')
        return entry

    def write(self, prompt_ids, trace_ids, halt_token_id, forced):
        atomic_json(self.path(prompt_ids), dict(fingerprint=self.fingerprint, prompt_ids=list(prompt_ids),
                    trace_ids=list(trace_ids), forced_halt=bool(forced)))
        self.read(prompt_ids, halt_token_id)


def halt_id(tokenizer):
    tokens = tokenizer.encode('</think>', add_special_tokens=False)
    if len(tokens) != 1:
        raise ValueError('ADOPD requires a single-token </think> delimiter')
    return tokens[0]


def training_prompts(config, tokenizer):
    """Yield original teacher IDs for prompts that fit the budgeted student template."""
    import pyarrow.parquet as pq
    from src.adopd_prompt import prompt_pair_ids
    files = config['train_files']
    if isinstance(files, str):
        files = [files]
    seen, visited = set(), 0
    limit = config.get('max_train_samples', 100000)
    for filename in files:
        for batch in pq.ParquetFile(filename).iter_batches(batch_size=128, columns=['prompt']):
            for row in batch.to_pylist():
                if visited >= limit:
                    return
                visited += 1
                tokens, student = prompt_pair_ids(tokenizer, row['prompt'], config['adopd_reasoning_budget'])
                if max(len(tokens), len(student)) > config['max_prompt_length']:
                    continue
                key = digest(tokens)
                if key not in seen:
                    seen.add(key)
                    yield tokens


def validate_training_cache(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['teacher_model_path'], local_files_only=True)
    cache = TraceCache(config['adopd_teacher_trace_cache'], cache_identity(config))
    stop = halt_id(tokenizer)
    prompts = list(training_prompts(config, tokenizer))
    if not prompts:
        raise ValueError('No valid training prompts for ADOPD')
    with ThreadPoolExecutor(max_workers=IO_THREADS) as pool:
        list(pool.map(lambda prompt: cache.read(prompt, stop), prompts))
    return len(prompts)


def prepare(config, dry_run=False):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['teacher_model_path'], local_files_only=True)
    stop = halt_id(tokenizer)
    cache = TraceCache(config['adopd_teacher_trace_cache'], cache_identity(config))

    def cached(prompt):
        try:
            cache.read(prompt, stop)
            return True
        except FileNotFoundError:
            return False

    prompts = list(training_prompts(config, tokenizer))
    with ThreadPoolExecutor(max_workers=IO_THREADS) as pool:
        found = list(pool.map(cached, prompts))
    missing = [prompt for prompt, hit in zip(prompts, found) if not hit]
    reused = sum(found)
    print(json.dumps(dict(teacher_cache=str(cache.root), reused=reused, missing=len(missing), dry_run=dry_run)), flush=True)
    if dry_run or not missing:
        return
    atomic_json(cache.root / 'manifest.json', cache.identity)
    nnodes = config.get('h100_nnodes', 1)
    gpus = config.get('n_gpus_per_node', config.get('teacher_tp_size', 1))
    tp = config.get('teacher_tp_size', 1)
    if gpus % tp:
        raise ValueError('Teacher tensor parallel size must divide n_gpus_per_node')
    if nnodes * gpus == tp:
        worker = TeacherCacheWorker(config)
        pending = deque(missing)
        try:
            while pending:
                prompts = [pending.popleft() for _ in range(min(len(pending), config['adopd_cache_batch_size']))]
                store(worker.cache, prompts, worker.generate(prompts), worker.stop)
                print(f'Teacher cache: {reused + len(missing) - len(pending)}/{reused + len(missing)}', flush=True)
        finally:
            worker.close()
        return
    prepare_distributed(config, missing, reused)


def store(cache, prompts, traces, halt_token_id):
    """Write one generated batch; each entry is its own file, so write them concurrently for slow storage."""
    if len(prompts) != len(traces):
        raise RuntimeError(f'Teacher returned {len(traces)} traces for {len(prompts)} prompts')

    def write(prompt, result):
        trace, forced = result
        cache.write(prompt, trace, halt_token_id, forced)

    with ThreadPoolExecutor(max_workers=IO_THREADS) as pool:
        list(pool.map(write, prompts, traces))


class TeacherCacheWorker:
    """One local tensor-parallel teacher replica, scheduled on a Ray GPU group."""

    def __init__(self, config):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.config = config
        self.cache = TraceCache(config['adopd_teacher_trace_cache'], cache_identity(config))
        self.tokenizer = AutoTokenizer.from_pretrained(config['teacher_model_path'], local_files_only=True)
        self.stop = halt_id(self.tokenizer)
        self.llm = LLM(model=config['teacher_model_path'], tokenizer=config['teacher_model_path'],
            tensor_parallel_size=config.get('teacher_tp_size', 1),
            distributed_executor_backend='mp', dtype='bfloat16',
            max_model_len=config['max_prompt_length'] + config['adopd_teacher_max_completion_length'],
            gpu_memory_utilization=0.85, enable_prefix_caching=True,
            enforce_eager=True, max_num_batched_tokens=4096, enable_chunked_prefill=True)

    def ready(self):
        import socket
        return {'node': socket.gethostname(), 'gpus': os.environ.get('CUDA_VISIBLE_DEVICES')}

    def generate(self, prompts):
        from vllm import SamplingParams
        config, stop = self.config, self.stop
        budget = config['adopd_teacher_max_completion_length']
        params = [SamplingParams(max_tokens=budget, temperature=1.0, top_p=1.0, top_k=-1,
                  stop_token_ids=[stop], seed=(config['adopd_cache_seed'] + int(digest(p)[:8], 16)) % (2**31))
                  for p in prompts]
        outputs = self.llm.generate([dict(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)

        def closed(output):
            trace = list(output.outputs[0].token_ids)
            forced = stop not in trace
            if not forced:
                trace = trace[:trace.index(stop) + 1]
            else:
                eos = {self.tokenizer.eos_token_id, self.tokenizer.bos_token_id}
                if trace and trace[-1] in eos:
                    trace.pop()
                trace = trace[:budget - 1] + [stop]
            return trace, forced

        # Returned, not written: a replica on another node cannot reach the driver's node-local cache.
        return [closed(output) for output in outputs]

    def close(self):
        # Shut down the engine (and its CUDA subprocesses) before training starts.
        engine = getattr(self.llm, 'llm_engine', None)
        if engine is not None:
            engine.engine_core.shutdown()
        self.llm = None


def cache_worker_layout(config, nodes):
    """Use every configured GPU, with each TP replica contained within one node."""
    nnodes, gpus = config.get('h100_nnodes', 1), config.get('n_gpus_per_node', 1)
    tp = config.get('teacher_tp_size', 1)
    if gpus % tp:
        raise ValueError('Teacher tensor parallel size must divide n_gpus_per_node')
    eligible = sorted((node for node in nodes if node['Alive']
                       and node['Resources'].get('GPU', 0) >= gpus), key=lambda node: node['NodeID'])
    if len(eligible) < nnodes:
        raise RuntimeError(f'Teacher cache needs {nnodes} nodes x {gpus} GPUs, '
                           f'but Ray has only {len(eligible)} matching nodes')
    return [node['NodeID'] for node in eligible[:nnodes] for _ in range(gpus // tp)]


def prepare_distributed(config, missing, reused):
    import ray
    from transformers import AutoTokenizer
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Attach to the allocation's Ray cluster, rather than creating a head-only
    # multiprocessing engine. A local Ray cluster also handles multi-replica nodes.
    ray.init(address=os.environ.get('RAY_ADDRESS'), runtime_env={'env_vars': {
        'PYTHONPATH': os.environ.get('PYTHONPATH', str(Path(__file__).resolve().parents[1])),
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
        # Training disables Ray's device masking; independent cache replicas
        # must see only their assigned GPU group.
        'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '0',
    }})
    # The driver's node holds the cache; every replica's traces are written here.
    cache = TraceCache(config['adopd_teacher_trace_cache'], cache_identity(config))
    stop = halt_id(AutoTokenizer.from_pretrained(config['teacher_model_path'], local_files_only=True))
    workers, active = [], {}
    initialized, succeeded = False, False
    total, pending = reused + len(missing), deque(missing)
    try:
        layout = cache_worker_layout(config, ray.nodes())
        remote_worker = ray.remote(num_cpus=1, num_gpus=config.get('teacher_tp_size', 1))(TeacherCacheWorker)
        for node in layout:
            workers.append(remote_worker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False)
            ).remote(config))
        locations = ray.get([worker.ready.remote() for worker in workers])
        initialized = True
        print(json.dumps({'teacher_cache_workers': locations,
                          'nnodes': config.get('h100_nnodes', 1),
                          'n_gpus_per_node': config['n_gpus_per_node']}), flush=True)
        idle = list(workers)
        last_report = 0.0
        while pending or active:
            for worker in list(idle):
                if not pending:
                    break
                prompts = [pending.popleft() for _ in range(min(len(pending), config['adopd_cache_batch_size']))]
                future = worker.generate.remote(prompts)
                active[future] = (worker, prompts)
                idle.remove(worker)
            if active:
                ready, _ = ray.wait(list(active), num_returns=1, timeout=5)
                for future in ready:
                    worker, prompts = active[future]
                    # Fails the preparation stage on a generation error.
                    store(cache, prompts, ray.get(future), stop)
                    del active[future]
                    idle.append(worker)
            if time.monotonic() - last_report >= 30 or not (pending or active):
                completed = total - len(pending) - sum(len(prompts) for _, prompts in active.values())
                print(f'Teacher cache: {completed}/{total} '
                      f'(active replicas={len(active)}, pending={len(pending)})', flush=True)
                last_report = time.monotonic()
        succeeded = True
    finally:
        try:
            if initialized and succeeded:
                ray.get([worker.close.remote() for worker in workers], timeout=60)
        finally:
            for worker in workers:
                ray.kill(worker, no_restart=True)
            ray.shutdown()
