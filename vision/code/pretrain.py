
from drawing import DrawingGame
from cnn import DoodleClassifier
import torch
import pytorch_lightning as pl
import pygame
from utils import render_multiline
from download_doodles import data_folder, load_data

# A broader range of classes for pretraining
classes = [
    "The Eiffel Tower",
    "airplane",
    "alarm clock",
    "axe",
    "bee",
    "bicycle",
    "birthday cake",
    "baseball",
    "butterfly",
    "cat",
    "circle",
    "dog",
    "dragon",
    "house",
    "monkey",
    "octopus",
    "stethoscope",
    "popsicle",
    "windmill"
]

if __name__ == "__main__":
    train_data, val_data, test_data = load_data(percent_data = 1.0, classes = classes)
    model = DoodleClassifier(classes = classes)
    trainer = pl.Trainer(max_epochs = 10,
                        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
                        devices = 1,
                        val_check_interval = 0.1,
                        callbacks = [pl.callbacks.TQDMProgressBar(), pl.callbacks.LearningRateMonitor(logging_interval = "step")]
                        )
    trainer.fit(model, train_dataloaders = train_data, val_dataloaders = val_data)
    # trainer.test(model, dataloaders = test_data)
    # Save the model
    trainer.save_checkpoint(f"{data_folder}/pretrained_model.ckpt")