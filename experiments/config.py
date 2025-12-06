from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml

@dataclass
class ModelConfig:
    name: str
    temperature: float = 0.7
    max_tokens: int = 2048

@dataclass
class InferenceConfig:
    num_gpus: int = 1
    use_ray: bool = False
    batch_size: int = 64
    max_num_seqs: int = 256
    max_model_len: int = 4096
    max_num_batched_tokens: int = 65536
    enable_chunked_prefill: bool = True

@dataclass
class InterventionConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self):
        if not self.params:
            return self.name
        param_str = '_'.join((f'{k}{v}' for k, v in sorted(self.params.items())))
        return f'{self.name}_{param_str}'

@dataclass
class ExperimentConfig:
    n_rollouts: int = 4
    batch_size: int = 64
    random_seed: int = 42
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model_a: ModelConfig = field(default_factory=lambda: ModelConfig(name='Qwen/Qwen3-4B', temperature=0.7, max_tokens=2048))
    model_b: ModelConfig = field(default_factory=lambda: ModelConfig(name='meta-llama/Llama-3.2-3B-Instruct', temperature=0.7, max_tokens=512))
    dataset_name: str = 'DigitalLearningGmbH/MATH-lighteval'
    dataset_split: str = 'test'
    max_examples: Optional[int] = None
    output_base_dir: Path = field(default_factory=lambda: Path('outputs'))
    truncate_first_k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    truncate_last_k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    truncate_contiguous_k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    truncate_p_values: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    filler_p_values: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    error_injection_enabled: bool = True
    wikipedia_subset: str = '20231101.en'
    wikipedia_num_articles: int = 1000

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        models = data.get('models', {})
        model_a_data = models.get('model_a', {})
        model_b_data = models.get('model_b', {})
        inference_data = data.get('inference', {})
        return cls(n_rollouts=data.get('n_rollouts', 4), batch_size=data.get('batch_size', 64), random_seed=data.get('random_seed', 42), inference=InferenceConfig(num_gpus=inference_data.get('num_gpus', 1), use_ray=inference_data.get('use_ray', False), batch_size=inference_data.get('batch_size', 64), max_num_seqs=inference_data.get('max_num_seqs', 256), max_model_len=inference_data.get('max_model_len', 4096), max_num_batched_tokens=inference_data.get('max_num_batched_tokens', 65536), enable_chunked_prefill=inference_data.get('enable_chunked_prefill', True)), model_a=ModelConfig(name=model_a_data.get('name', 'Qwen/Qwen3-4B'), temperature=model_a_data.get('temperature', 0.7), max_tokens=model_a_data.get('max_tokens', 2048)), model_b=ModelConfig(name=model_b_data.get('name', 'meta-llama/Llama-3.2-3B-Instruct'), temperature=model_b_data.get('temperature', 0.7), max_tokens=model_b_data.get('max_tokens', 512)), dataset_name=data.get('dataset', {}).get('name', 'DigitalLearningGmbH/MATH-lighteval'), dataset_split=data.get('dataset', {}).get('split', 'test'), max_examples=data.get('dataset', {}).get('max_examples'), output_base_dir=Path(data.get('output', {}).get('base_dir', 'outputs')), truncate_first_k_values=data.get('interventions', {}).get('truncate_first', {}).get('k_values', [1, 2, 3, 5]), truncate_last_k_values=data.get('interventions', {}).get('truncate_last', {}).get('k_values', [1, 2, 3, 5]), truncate_contiguous_k_values=data.get('interventions', {}).get('truncate_contiguous', {}).get('k_values', [1, 2, 3, 5]), truncate_p_values=data.get('interventions', {}).get('truncate_percent', {}).get('p_values', [0.1, 0.2, 0.3, 0.5]), filler_p_values=data.get('interventions', {}).get('filler_replacement', {}).get('p_values', [0.1, 0.2, 0.3, 0.5]), error_injection_enabled=data.get('interventions', {}).get('error_injection', {}).get('enabled', True), wikipedia_subset=data.get('interventions', {}).get('filler_replacement', {}).get('wikipedia_subset', '20231101.en'), wikipedia_num_articles=data.get('interventions', {}).get('filler_replacement', {}).get('num_articles', 1000))

    def get_intervention_configs(self):
        configs = []
        for k in self.truncate_first_k_values:
            configs.append(InterventionConfig('truncate_first', {'k': k}))
        for k in self.truncate_last_k_values:
            configs.append(InterventionConfig('truncate_last', {'k': k}))
        for k in self.truncate_contiguous_k_values:
            configs.append(InterventionConfig('truncate_contiguous', {'k': k}))
        for p in self.truncate_p_values:
            configs.append(InterventionConfig('truncate_percent', {'p': p}))
        if self.error_injection_enabled:
            configs.append(InterventionConfig('error_injection', {}))
        for p in self.filler_p_values:
            configs.append(InterventionConfig('filler_replacement', {'p': p}))
        return configs

    @property
    def baselines_dir(self):
        return self.output_base_dir / 'baselines'

    @property
    def interventions_dir(self):
        return self.output_base_dir / 'interventions'
