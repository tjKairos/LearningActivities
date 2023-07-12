from drawing import DrawingGame
from cnn import load_data, DoodleClassifier
from download_doodles import labeled_classes
from pretrain import classes as pretrain_classes
from finetune import finetuned_model
import torch
import pytorch_lightning as pl
import pygame
import numpy as np
from utils import render_multiline

classes = pretrain_classes

if __name__ == "__main__":
    fine_tune = False
    just_labeled = False
    model = finetuned_model(do_tune = fine_tune)
    model.hparams.classes = pretrain_classes
    if just_labeled:
        classes = labeled_classes(pretrain_classes)

    game = DrawingGame(width = 800, height = 600, resolution = 28, blur = 0.4)

    font = pygame.font.Font("freesansbold.ttf", 24)

    update_every = 10
    step = 0
    text = font.render("Prediction", True, (255, 255, 255), (0, 0, 0))
    probabilities_text = [""]
    longest_class_name = max(classes, key = len)

    for events in game:
        # Render the text to the screen
        if step % update_every == 0:
            image = game.drawing.get_image()
            tensor = torch.from_numpy(image).float().unsqueeze(0)[:, :, :, 0]
            prediction = model.predict(tensor, classes = classes)
            prediction_probs = model.predict_probs(tensor, classes = classes)
            # Sort predictions by probability
            predictions = sorted(zip(classes, prediction_probs[0]), key = lambda x: x[1], reverse = True)
            # Take the 5 most probable classes and format them as strings
            probabilities = [f"{p[0]}: {p[1]:.1%}" for p in predictions]
            probabilities = probabilities[:5]

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c or event.key == pygame.K_SPACE:
                    game.drawing.pixels = np.zeros((game.drawing.width, game.drawing.height, 3), dtype = np.int16)

        text_height = font.get_height() * len(probabilities) + 2
        text_width = font.size(longest_class_name + ": 00.0%")[0]
        render_multiline(probabilities, font, (255, 255, 255), (0, 0, 0), game.width - text_width, game.height - text_height, screen = game.screen)
        step += 1