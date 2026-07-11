from abc import ABC, abstractmethod
from typing import Any, Dict

from ..config import PipelineConfig


class PhaseHandler(ABC):
    @abstractmethod
    def execute(self, state: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
        pass
