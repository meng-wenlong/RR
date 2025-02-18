from abc import ABC, abstractmethod
from datasets import Dataset


class AbstractCandidateSelector(ABC):
    @abstractmethod
    def select_candidates(self, test_dataset: Dataset, candidates):
        pass
