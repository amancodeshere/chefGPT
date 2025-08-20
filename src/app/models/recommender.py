from abc import abstractclassmethod, abstractmethod

import torch
from torch.xpu import device


class Recommender:
    """
    Here we set up the abstract recommender super class
    """
    def __init__(self):
        self.device = device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def recommend(self, user_id, top_k=10):
        pass