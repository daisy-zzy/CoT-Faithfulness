from typing import List
from vllm import LLM, SamplingParams

class VLLMEngine:
    def __init__(
        self,
        model_name: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: List[str] | None = None,
    ) -> List[str]:
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
