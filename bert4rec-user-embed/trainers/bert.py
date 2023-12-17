from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch
import torch.nn as nn


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, labels = batch
        batch_idx = seqs[:, 0]
        seqs = seqs[:, 1:]
        labels = labels[:, 1:]
        # print("trainers/bert------input shape", seqs.shape, batch_idx.shape, labels.shape)
        logits = self.model(seqs, batch_idx)  # B x T x V
        # print("trainers/bert------output shape", logits.shape)

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        # print("trainers/bert------ce shape", logits.shape, labels.shape)
        labels = labels.reshape(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        # print("trainers/bert calculate metrics------ seqs shape", seqs.shape, seqs[0].shape, seqs[0][1:].shape)
        batch_idx = seqs[:, 0]
        seqs = seqs[:, 1:]
        candidates = candidates[:, 1:]
        labels = labels[:, 1:]

        scores = self.model(seqs, batch_idx)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics
