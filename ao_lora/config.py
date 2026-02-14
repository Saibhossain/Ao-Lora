from dataclasses import dataclass, field
from typing import List


@dataclass
class AOLoRAConfig:

    r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])

    tau: float = 0.15
    lambda_ao: float = 0.01