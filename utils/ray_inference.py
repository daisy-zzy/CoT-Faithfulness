"""
Ray-based data parallel inference for multi-GPU setups.

This module provides a Ray Data + vLLM integration for running
inference across multiple GPUs with data parallelism.
"""

import os
import sys

# Fix uvloop/asyncio compatibility with Ray
# Must set this BEFORE importing asyncio if uvloop is installed
os.environ.setdefault("RAY_USE_UVLOOP", "0")  # Disable uvloop in Ray workers

import asyncio


# Create an event loop before any Ray operations
# This fixes "There is no current event loop in thread 'MainThread'" error
def _ensure_event_loop():
    """Ensure an event loop exists in the current thread."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


_ensure_event_loop()

import ray
from ray.data import DataContext
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from typing import List, Dict, Any, Optional
import pandas as pd


def init_ray(
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    tmp_dir: Optional[str] = None,
    timeout: int = 1800,
):
    """Initialize Ray with appropriate settings."""
    # Ensure event loop exists before Ray operations
    _ensure_event_loop()

    if tmp_dir is None:
        tmp_dir = os.environ.get(
            "RAY_TMPDIR", f"/data/user_data/{os.getenv('USER', 'user')}/ray_tmp"
        )
    os.makedirs(tmp_dir, exist_ok=True)

    kwargs = {"_temp_dir": tmp_dir}
    if num_cpus is not None:
        kwargs["num_cpus"] = num_cpus
    if num_gpus is not None:
        kwargs["num_gpus"] = num_gpus

    # Set runtime environment to propagate necessary env vars to workers
    kwargs["runtime_env"] = {
        "env_vars": {
            "MKL_THREADING_LAYER": "GNU",  # Fix MKL/libgomp incompatibility
            "MKL_SERVICE_FORCE_INTEL": "0",
        }
    }

    ray.init(**kwargs)

    DataContext.get_current().wait_for_min_actors_s = timeout


def shutdown_ray():
    """Shutdown Ray."""
    ray.shutdown()


class RayVLLMInference:
    """
    Ray-based vLLM inference engine for data parallel multi-GPU inference.

    This uses Ray Data's LLM processor to distribute inference across
    multiple GPUs, with each GPU running its own vLLM engine instance.
    """

    def __init__(
        self,
        model_name: str,
        num_gpus: int = 1,
        max_model_len: int = 4096,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        batch_size: int = 32,
        max_concurrent_batches: int = 32,
        max_num_batched_tokens: int = 65536,
        enable_chunked_prefill: bool = True,
        num_blocks: int = 64,
    ):
        """
        Initialize Ray vLLM inference engine.

        Args:
            model_name: HuggingFace model name
            num_gpus: Number of GPUs for data parallelism
            max_model_len: Maximum sequence length
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            batch_size: Batch size per GPU
            max_concurrent_batches: Max concurrent batches
            max_num_batched_tokens: Max batched tokens for continuous batching
            enable_chunked_prefill: Enable chunked prefill
            num_blocks: Number of data blocks for partitioning
        """
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
        # Auto-calculate num_blocks based on data size and GPUs
        # More blocks = better parallelism, but too many = overhead
        # Aim for ~256-512 rows per block minimum
        self.num_blocks = num_blocks

        # Calculate resources per engine
        import multiprocessing

        total_cpus = multiprocessing.cpu_count()
        self.engine_cpu_resources = max(1, total_cpus // num_gpus)

    def generate(
        self,
        prompts: List[str],
        seeds: Optional[List[int]] = None,
        extra_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for a list of prompts using Ray Data.

        Args:
            prompts: List of prompts to generate
            seeds: Optional list of seeds for reproducibility
            extra_data: Optional list of dicts with extra data to preserve

        Returns:
            List of dicts with 'prompt', 'output', and any extra_data fields
        """
        # Build dataframe
        data = {"prompt": prompts}
        if seeds is not None:
            data["seed"] = seeds
        if extra_data is not None:
            # Merge extra data columns
            for key in extra_data[0].keys():
                data[key] = [d[key] for d in extra_data]

        df = pd.DataFrame(data)

        # Auto-calculate optimal number of blocks
        # Want enough blocks to keep all GPUs busy, but not so many that overhead dominates
        # Rule: aim for ~500-1000 rows per block, minimum num_gpus * 4 blocks
        auto_num_blocks = max(self.num_gpus * 4, len(prompts) // 500)
        actual_num_blocks = (
            min(auto_num_blocks, self.num_blocks)
            if self.num_blocks
            else auto_num_blocks
        )

        # Create Ray dataset
        ds = ray.data.from_pandas(df).repartition(actual_num_blocks)

        print(
            f"Processing {len(prompts)} prompts across {self.num_gpus} GPUs ({actual_num_blocks} blocks)"
        )

        # Configure vLLM processor
        # Ray 2.52+ uses num_cpus/num_gpus instead of CPU/GPU in resources_per_bundle
        config = vLLMEngineProcessorConfig(
            model_source=self.model_name,
            resources_per_bundle={
                "num_cpus": self.engine_cpu_resources,
                "num_gpus": 1,
            },
            concurrency=self.num_gpus,
            engine_kwargs={
                "tensor_parallel_size": 1,  # No tensor parallelism, pure data parallel
                "enable_chunked_prefill": self.enable_chunked_prefill,
                "max_model_len": self.max_model_len,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "swap_space": 16,
                "trust_remote_code": True,
            },
            runtime_env={
                "env_vars": {
                    "MKL_THREADING_LAYER": "GNU",  # Fix MKL/libgomp incompatibility
                    "MKL_SERVICE_FORCE_INTEL": "0",
                    "HF_HOME": os.environ.get("HF_HOME", ""),
                    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                    "HUGGING_FACE_HUB_TOKEN": os.environ.get(
                        "HUGGING_FACE_HUB_TOKEN", os.environ.get("HF_TOKEN", "")
                    ),
                }
            },
            max_concurrent_batches=self.max_concurrent_batches,
            batch_size=self.batch_size,
            apply_chat_template=True,  # Let Ray handle chat template
            tokenize=True,
            detokenize=True,
        )

        # Preprocessing function - format as chat messages
        def preprocess(row):
            sampling_params = {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if "seed" in row and row["seed"] is not None:
                sampling_params["seed"] = int(row["seed"])

            return {
                "messages": [{"role": "user", "content": row["prompt"]}],
                "sampling_params": sampling_params,
            }

        # Postprocessing function
        def postprocess(row):
            result = {"output": row["generated_text"]}
            # Preserve all original columns
            for key in row.keys():
                if key not in ["generated_text", "messages", "sampling_params"]:
                    result[key] = row[key]
            return result

        # Build and run processor
        processor = build_llm_processor(
            config,
            preprocess=preprocess,
            postprocess=postprocess,
        )

        result_ds = processor(ds).materialize()

        # Collect results
        results = result_ds.to_pandas().to_dict("records")

        return results


def generate_with_ray(
    prompts: List[str],
    model_name: str,
    num_gpus: int = 1,
    seeds: Optional[List[int]] = None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    **kwargs,
) -> List[str]:
    """
    Convenience function for Ray-based generation.

    Returns just the output texts (not full records).
    """
    engine = RayVLLMInference(
        model_name=model_name,
        num_gpus=num_gpus,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )

    results = engine.generate(prompts, seeds=seeds)

    # Sort by original order if needed and return outputs
    # Ray may return results out of order, so we need to track indices
    return [r["output"] for r in results]
