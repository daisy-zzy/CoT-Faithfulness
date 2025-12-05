"""Experiment configuration management."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""

    num_gpus: int = 1
    use_ray: bool = False  # Use Ray for multi-GPU data parallelism
    batch_size: int = 64
    max_num_seqs: int = 256
    max_model_len: int = 4096
    max_num_batched_tokens: int = 65536
    enable_chunked_prefill: bool = True


@dataclass
class InterventionConfig:
    """Configuration for a single intervention run."""

    name: str  # e.g., "truncate_first"
    params: Dict[str, Any] = field(default_factory=dict)  # e.g., {"k": 3}

    @property
    def run_id(self) -> str:
        """Unique identifier for this intervention config."""
        if not self.params:
            return self.name
        param_str = "_".join(f"{k}{v}" for k, v in sorted(self.params.items()))
        return f"{self.name}_{param_str}"


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    n_rollouts: int = 4
    batch_size: int = 64
    random_seed: int = 42

    # Inference settings
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    model_a: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="Qwen/Qwen3-4B", temperature=0.7, max_tokens=2048
        )
    )
    model_b: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            name="meta-llama/Llama-3.2-3B-Instruct", temperature=0.7, max_tokens=512
        )
    )

    dataset_name: str = "DigitalLearningGmbH/MATH-lighteval"
    dataset_split: str = "test"
    max_examples: Optional[int] = None

    output_base_dir: Path = field(default_factory=lambda: Path("outputs"))

    # Intervention parameters - separate k_values for each truncate type
    truncate_first_k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    truncate_last_k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    truncate_contiguous_k_values: List[int] = field(
        default_factory=lambda: [1, 2, 3, 5]
    )
    truncate_p_values: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    filler_p_values: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    error_injection_enabled: bool = True  # Can disable to skip error injection

    # Wikipedia settings for filler text
    wikipedia_subset: str = "20231101.en"
    wikipedia_num_articles: int = 1000

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        models = data.get("models", {})
        model_a_data = models.get("model_a", {})
        model_b_data = models.get("model_b", {})
        inference_data = data.get("inference", {})

        return cls(
            n_rollouts=data.get("n_rollouts", 4),
            batch_size=data.get("batch_size", 64),
            random_seed=data.get("random_seed", 42),
            inference=InferenceConfig(
                num_gpus=inference_data.get("num_gpus", 1),
                use_ray=inference_data.get("use_ray", False),
                batch_size=inference_data.get("batch_size", 64),
                max_num_seqs=inference_data.get("max_num_seqs", 256),
                max_model_len=inference_data.get("max_model_len", 4096),
                max_num_batched_tokens=inference_data.get(
                    "max_num_batched_tokens", 65536
                ),
                enable_chunked_prefill=inference_data.get(
                    "enable_chunked_prefill", True
                ),
            ),
            model_a=ModelConfig(
                name=model_a_data.get("name", "Qwen/Qwen3-4B"),
                temperature=model_a_data.get("temperature", 0.7),
                max_tokens=model_a_data.get("max_tokens", 2048),
            ),
            model_b=ModelConfig(
                name=model_b_data.get("name", "meta-llama/Llama-3.2-3B-Instruct"),
                temperature=model_b_data.get("temperature", 0.7),
                max_tokens=model_b_data.get("max_tokens", 512),
            ),
            dataset_name=data.get("dataset", {}).get(
                "name", "DigitalLearningGmbH/MATH-lighteval"
            ),
            dataset_split=data.get("dataset", {}).get("split", "test"),
            max_examples=data.get("dataset", {}).get("max_examples"),
            output_base_dir=Path(data.get("output", {}).get("base_dir", "outputs")),
            truncate_first_k_values=data.get("interventions", {})
            .get("truncate_first", {})
            .get("k_values", [1, 2, 3, 5]),
            truncate_last_k_values=data.get("interventions", {})
            .get("truncate_last", {})
            .get("k_values", [1, 2, 3, 5]),
            truncate_contiguous_k_values=data.get("interventions", {})
            .get("truncate_contiguous", {})
            .get("k_values", [1, 2, 3, 5]),
            truncate_p_values=data.get("interventions", {})
            .get("truncate_percent", {})
            .get("p_values", [0.1, 0.2, 0.3, 0.5]),
            filler_p_values=data.get("interventions", {})
            .get("filler_replacement", {})
            .get("p_values", [0.1, 0.2, 0.3, 0.5]),
            error_injection_enabled=data.get("interventions", {})
            .get("error_injection", {})
            .get("enabled", True),
            wikipedia_subset=data.get("interventions", {})
            .get("filler_replacement", {})
            .get("wikipedia_subset", "20231101.en"),
            wikipedia_num_articles=data.get("interventions", {})
            .get("filler_replacement", {})
            .get("num_articles", 1000),
        )

    def get_intervention_configs(self) -> List[InterventionConfig]:
        """Generate all intervention configurations to run."""
        configs = []

        # 1a: Truncate first k
        for k in self.truncate_first_k_values:
            configs.append(InterventionConfig("truncate_first", {"k": k}))

        # 1b: Truncate last k
        for k in self.truncate_last_k_values:
            configs.append(InterventionConfig("truncate_last", {"k": k}))

        # 1c: Truncate contiguous k
        for k in self.truncate_contiguous_k_values:
            configs.append(InterventionConfig("truncate_contiguous", {"k": k}))

        # 1d: Truncate percent
        for p in self.truncate_p_values:
            configs.append(InterventionConfig("truncate_percent", {"p": p}))

        # 2: Error injection (can be disabled via config)
        if self.error_injection_enabled:
            configs.append(InterventionConfig("error_injection", {}))

        # 3: Filler replacement
        for p in self.filler_p_values:
            configs.append(InterventionConfig("filler_replacement", {"p": p}))

        return configs

    @property
    def baselines_dir(self) -> Path:
        return self.output_base_dir / "baselines"

    @property
    def interventions_dir(self) -> Path:
        return self.output_base_dir / "interventions"
