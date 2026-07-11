from abc import ABC, abstractmethod
from typing import Any

from ..config import PipelineConfig


class PhaseHandler(ABC):
    @abstractmethod
    def execute(self, state: dict, config: PipelineConfig) -> dict:
        pass
