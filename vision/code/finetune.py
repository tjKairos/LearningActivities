
from drawing import DrawingGame
from cnn import load_data, DoodleClassifier
import torch
import pytorch_lightning as pl
import pygame
from utils import render_multiline
from download_doodles import data_folder, load_labeled
from pretrain import classes as pretrain_classes


def finetuned_model(do_tune: bool = True):
    model = DoodleClassifier.load_from_checkpoint(f"{data_folder}/pretrained_model.ckpt", map_location = torch.device("cpu"))
    if not do_tune:
        return model
    train_data, val_data, test_data = load_labeled(classes = pretrain_classes, show = False)
    trainer = pl.Trainer(max_epochs = 20,
                        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
                        devices = 1,
                        val_check_interval = 0.1,
                        log_every_n_steps = 1,
                        callbacks = [pl.callbacks.TQDMProgressBar(), pl.callbacks.LearningRateMonitor(logging_interval = "step")]
                        )
    trainer.validate(model, dataloaders = val_data)
    trainer.fit(model, train_dataloaders = train_data, val_dataloaders = val_data)
    trainer.validate(model, dataloaders = val_data)
    return model

if __name__ == "__main__":
    finetuned_model()
    # trainer.test(model, dataloaders = test_data)
    # Save the model
    # trainer.save_checkpoint(f"{data_folder}/pretrained_model.ckpt")