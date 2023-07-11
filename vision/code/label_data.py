import os
import pygame
import numpy as np
from download_doodles import data_folder
from drawing import DrawingGame


print("Labeling more data for ...")
label = input("Label: ")

game = DrawingGame(width = 1000, height = 800, resolution = 28, blur = 0.4)
labeled = np.zeros((0, 28, 28, 3), dtype = np.uint8)

for events in game:
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN or event.key == pygame.K_s or event.key == pygame.K_SPACE:
                labeled = np.concatenate((labeled, game.drawing.get_image().reshape(1, 28, 28, 3)), axis = 0)
                print("Saved!")
                # Clear the screen
                game.drawing.pixels = np.zeros((game.drawing.width, game.drawing.height, 3), dtype = np.int16)
            elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                pygame.quit()

# Save labeled data
if not os.path.exists(f"{data_folder}/{label}"):
    os.makedirs(f"{data_folder}/{label}")
id = 1
while os.path.exists(f"{data_folder}/{label}/{id:03}.npy"):
    id += 1
np.save(f"{data_folder}/{label}/{id:03}.npy", labeled)