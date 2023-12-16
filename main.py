import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()

def investigate():
    export_root = setup_train(args)
    print("******setup finished******")
    train_loader, val_loader, test_loader = dataloader_factory(args)
    print("******data loaded into dataloaders******")
    for batch_idx, batch in enumerate(train_loader):
        seqs, labels = batch

        for seq in seqs:
            print(seq)
            print("-------")
            print(labels[0])
            break
        break

if __name__ == '__main__':
    if args.mode == 'train':
        # investigate()
        train()
    else:
        raise ValueError('Invalid mode')
