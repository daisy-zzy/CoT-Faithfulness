from typing import List, Optional
from vllm import LLM, SamplingParams


class VLLMEngine:
    """
    Wrapper around vLLM for efficient batched inference.

    Key optimizations:
    - Uses chunked prefill for better memory utilization
    - Configurable batching for high throughput
    - Supports seed per request for reproducible rollouts
    - Tensor parallelism for multi-GPU inference
    """

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        max_num_seqs: int = 256,  # Max concurrent sequences
        enable_chunked_prefill: bool = True,
        max_num_batched_tokens: int = 32768,  # Higher for better throughput
        swap_space: int = 4,  # GiB of CPU swap space per GPU
    ):
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_num_seqs=max_num_seqs,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_batched_tokens=max_num_batched_tokens,
            swap_space=swap_space,
        )

    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
    ) -> List[str]:
        """
        Generate completions for a batch of prompts.

        Args:
            prompts: List of prompts to generate completions for
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            max_tokens: Maximum tokens to generate per prompt
            stop: Stop sequences
            seeds: Optional list of seeds (one per prompt) for reproducibility

        Returns:
            List of generated text completions
        """
        # If seeds provided, create per-request sampling params
        if seeds is not None and len(seeds) == len(prompts):
            # Use request-level seeds for reproducible rollouts
            sampling_params_list = [
                SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    seed=seed,
                )
                for seed in seeds
            ]
            outputs = self.llm.generate(prompts, sampling_params=sampling_params_list)
        else:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
            )
            outputs = self.llm.generate(prompts, sampling_params)

        texts = []
        for out in outputs:
            # first candidate
            texts.append(out.outputs[0].text)
        return texts

    def generate_batched(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Generate completions with explicit batching and progress bar.

        This is useful for very large prompt lists where you want
        to see progress and potentially checkpoint intermediate results.
        """
        from tqdm import tqdm

        all_results = []
        iterator = range(0, len(prompts), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Generating",
                total=(len(prompts) + batch_size - 1) // batch_size,
            )

        for i in iterator:
            batch_prompts = prompts[i : i + batch_size]
            batch_seeds = seeds[i : i + batch_size] if seeds else None

            batch_outputs = self.generate(
                batch_prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                seeds=batch_seeds,
            )
            all_results.extend(batch_outputs)

        return all_results
