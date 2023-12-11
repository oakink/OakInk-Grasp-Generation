from abc import ABCMeta, abstractmethod
import torch.nn as nn


class ModelABC(nn.Module, metaclass=ABCMeta):

    def __init__(self) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.has_loss = False
        self.has_eval = False

    def setup(self, summary_writer, log_freq, **kwargs):
        self.summary = summary_writer
        self.log_freq = log_freq
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _build_loss(self, **kwargs):
        self.has_loss = True

    def _build_evaluation(self, **kwargs):
        self.has_eval = True

    def training_step(self):
        pass

    def on_train_finished(self):
        pass

    def validation_step(self):
        pass

    def on_val_finished(self):
        pass

    def compute_loss(self):
        pass

    def testing_step(self, batch, batch_idx):
        pass

    def inference_step(self, batch, batch_idx):
        pass
