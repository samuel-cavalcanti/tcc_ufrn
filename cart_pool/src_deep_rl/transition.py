from dataclasses import dataclass
from typing import Optional
import torch



@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    next_state: Optional[torch.Tensor]
    reward: torch.Tensor


