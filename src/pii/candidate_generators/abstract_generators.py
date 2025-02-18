from abc import ABC, abstractmethod
from datasets import Dataset


class AbstratCandidateGenerator(ABC):
    @abstractmethod
    def generate_candidates(self, test_dataset: Dataset) -> list[dict]:
        pass
