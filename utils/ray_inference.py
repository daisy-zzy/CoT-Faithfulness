import os
import sys
os.environ.setdefault('RAY_USE_UVLOOP', '0')
import asyncio

def _ensure_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError('Event loop is closed')
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
_ensure_event_loop()
import ray
from ray.data import DataContext
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
import pandas as pd

def init_ray(num_cpus=None, num_gpus=None, tmp_dir=None, timeout=1800):
    _ensure_event_loop()
    if tmp_dir is None:
        tmp_dir = os.environ.get('RAY_TMPDIR', f"/data/user_data/{os.getenv('USER', 'user')}/ray_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    kwargs = {'_temp_dir': tmp_dir}
    if num_cpus is not None:
        kwargs['num_cpus'] = num_cpus
    if num_gpus is not None:
        kwargs['num_gpus'] = num_gpus
    kwargs['runtime_env'] = {'env_vars': {'MKL_THREADING_LAYER': 'GNU', 'MKL_SERVICE_FORCE_INTEL': '0'}}
    ray.init(**kwargs)
    DataContext.get_current().wait_for_min_actors_s = timeout

def shutdown_ray():
    ray.shutdown()

class RayVLLMInference:

    def __init__(self, model_name, num_gpus=1, max_model_len=4096, max_tokens=2048, temperature=0.7, top_p=0.95, batch_size=32, max_concurrent_batches=32, max_num_batched_tokens=65536, enable_chunked_prefill=True, num_blocks=64):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.num_blocks = num_blocks
        import multiprocessing
        total_cpus = multiprocessing.cpu_count()
        self.engine_cpu_resources = max(1, total_cpus // num_gpus)

    def generate(self, prompts, seeds=None, extra_data=None):
        data = {'prompt': prompts}
        if seeds is not None:
            data['seed'] = seeds
        if extra_data is not None:
            for key in extra_data[0].keys():
                data[key] = [d[key] for d in extra_data]
        df = pd.DataFrame(data)
        auto_num_blocks = max(self.num_gpus * 4, len(prompts) // 500)
        actual_num_blocks = min(auto_num_blocks, self.num_blocks) if self.num_blocks else auto_num_blocks
        ds = ray.data.from_pandas(df).repartition(actual_num_blocks)
        print(f'Processing {len(prompts)} prompts across {self.num_gpus} GPUs ({actual_num_blocks} blocks)')
        config = vLLMEngineProcessorConfig(model_source=self.model_name, resources_per_bundle={'num_cpus': self.engine_cpu_resources, 'num_gpus': 1}, concurrency=self.num_gpus, engine_kwargs={'tensor_parallel_size': 1, 'enable_chunked_prefill': self.enable_chunked_prefill, 'max_model_len': self.max_model_len, 'max_num_batched_tokens': self.max_num_batched_tokens, 'swap_space': 16, 'trust_remote_code': True}, runtime_env={'env_vars': {'MKL_THREADING_LAYER': 'GNU', 'MKL_SERVICE_FORCE_INTEL': '0', 'HF_HOME': os.environ.get('HF_HOME', ''), 'HF_TOKEN': os.environ.get('HF_TOKEN', ''), 'HUGGING_FACE_HUB_TOKEN': os.environ.get('HUGGING_FACE_HUB_TOKEN', os.environ.get('HF_TOKEN', ''))}}, max_concurrent_batches=self.max_concurrent_batches, batch_size=self.batch_size, apply_chat_template=True, tokenize=True, detokenize=True)

        def preprocess(row):
            sampling_params = {'max_tokens': self.max_tokens, 'temperature': self.temperature, 'top_p': self.top_p}
            if 'seed' in row and row['seed'] is not None:
                sampling_params['seed'] = int(row['seed'])
            return {'messages': [{'role': 'user', 'content': row['prompt']}], 'sampling_params': sampling_params}

        def postprocess(row):
            result = {'output': row['generated_text']}
            for key in row.keys():
                if key not in ['generated_text', 'messages', 'sampling_params']:
                    result[key] = row[key]
            return result
        processor = build_llm_processor(config, preprocess=preprocess, postprocess=postprocess)
        result_ds = processor(ds).materialize()
        results = result_ds.to_pandas().to_dict('records')
        return results

def generate_with_ray(prompts, model_name, num_gpus=1, seeds=None, max_tokens=2048, temperature=0.7, **kwargs):
    engine = RayVLLMInference(model_name=model_name, num_gpus=num_gpus, max_tokens=max_tokens, temperature=temperature, **kwargs)
    results = engine.generate(prompts, seeds=seeds)
    return [r['output'] for r in results]