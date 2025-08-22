from abc import abstractmethod, ABC

import torch
from torch import device


class Recommender(ABC):
    """
    Here we set up the abstract recommender super class
    """
    def __init__(self):
        self.device = device(Recommender._select_device(self))

    @staticmethod
    def _select_device(self) -> str:
        """
        Chooses an available accelerator in a deterministic order.
        Returns a string like 'cuda', 'mps', or 'cpu'.
        """
        if torch is None:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

