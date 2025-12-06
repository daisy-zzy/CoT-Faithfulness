from vllm import LLM, SamplingParams

class VLLMEngine:

    def __init__(self, model_name, max_model_len=4096, gpu_memory_utilization=0.9, tensor_parallel_size=1, dtype='bfloat16', trust_remote_code=True, max_num_seqs=256, enable_chunked_prefill=True, max_num_batched_tokens=32768, swap_space=4):
        self.model_name = model_name
        self.llm = LLM(model=model_name, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size, dtype=dtype, trust_remote_code=trust_remote_code, max_num_seqs=max_num_seqs, enable_chunked_prefill=enable_chunked_prefill, max_num_batched_tokens=max_num_batched_tokens, swap_space=swap_space)

    def generate(self, prompts, temperature=0.7, top_p=0.95, max_tokens=512, stop=None, seeds=None):
        if seeds is not None and len(seeds) == len(prompts):
            sampling_params_list = [SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop, seed=seed) for seed in seeds]
            outputs = self.llm.generate(prompts, sampling_params=sampling_params_list)
        else:
            sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop)
            outputs = self.llm.generate(prompts, sampling_params)
        texts = []
        for out in outputs:
            texts.append(out.outputs[0].text)
        return texts

    def generate_batched(self, prompts, temperature=0.7, top_p=0.95, max_tokens=512, stop=None, seeds=None, batch_size=64, show_progress=True):
        from tqdm import tqdm
        all_results = []
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc='Generating', total=(len(prompts) + batch_size - 1) // batch_size)
        for i in iterator:
            batch_prompts = prompts[i:i + batch_size]
            batch_seeds = seeds[i:i + batch_size] if seeds else None
            batch_outputs = self.generate(batch_prompts, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop, seeds=batch_seeds)
            all_results.extend(batch_outputs)
        return all_results