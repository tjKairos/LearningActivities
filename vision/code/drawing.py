import pygame
import numpy as np
import cv2


class DrawingGame:
    screen: pygame.Surface
    drawing: "Drawing"
    width: int
    height: int
    scale: int = 1
    blur: float = 0.0

    def __init__(self, width: int, height: int, resolution = 28, grayscale = True, scale: int = 1, blur: float = 0.0):
        self.width = width
        self.height = height
        self.scale = scale
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        # Create a Drawing object
        self.drawing = Drawing(self.screen, resolution, resolution, grayscale = grayscale, blur = blur)

    def __iter__(self):
        return self.drawing.__iter__()


class Drawing:
    screen: pygame.Surface
    screen_pixels: pygame.PixelArray
    pixels: np.ndarray
    surface: pygame.Surface
    width: int
    height: int
    grayscale: bool
    x_scale: float
    y_scale: float
    pressed: bool = False
    blur: float = 0.0

    def __init__(self, screen: pygame.Surface, width: int, height: int, grayscale = True, blur: float = 0.0):
        self.screen = screen
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.blur = blur

        self.surface = pygame.Surface((width, height))
        self.screen_pixels = pygame.PixelArray(self.surface)
        self.pixels = np.zeros((width, height, 3), dtype=np.int16) # uint16 for overflow
        screen_shape = screen.get_rect().size
        self.x_scale = screen_shape[0] / width
        self.y_scale = screen_shape[1] / height

    def draw(self, x: int, y: int, color: tuple[int, int, int] = None):
        if color is None:
            color = (255, 255, 255)

        x, y = int(y // self.x_scale), int(x // self.y_scale)
        blur_window = 5
        if self.blur > 0:
            blurred = np.zeros((blur_window, blur_window, 3), dtype = np.int16)
            blurred[blur_window // 2, blur_window // 2, :] = color
            blurred = cv2.GaussianBlur(blurred, (blur_window, blur_window), self.blur)
            l_width = blur_window // 2
            if x - l_width < 0:
                l_width = x
            r_width = blur_window // 2
            if x + r_width >= self.width:
                r_width = self.width - x - 1
            t_width = blur_window // 2
            if y - t_width < 0:
                t_width = y
            b_width = blur_window // 2
            if y + b_width >= self.height:
                b_width = self.height - y - 1
            blurred_clamped = blurred[max(0, blur_window // 2 - l_width):min(blur_window, blur_window // 2 + r_width + 1),
                                      max(0, blur_window // 2 - t_width):min(blur_window, blur_window // 2 + b_width + 1), :]
            if color == (0, 0, 0):
                blurred_clamped[:, :, :] = 10
                self.pixels[x - l_width:x + r_width + 1, y - t_width:y + b_width + 1, :] -= blurred_clamped
            else:
                self.pixels[x - l_width:x + r_width + 1, y - t_width:y + b_width + 1, :] += blurred_clamped

        self.pixels[x, y, :] = color
        self.pixels[self.pixels > 255] = 255
        self.pixels[self.pixels < 0] = 0

    def render(self):
        pygame.surfarray.blit_array(self.surface, self.pixels)
        self.screen.blit(pygame.transform.scale(self.surface, self.screen.get_rect().size), (0, 0))

    def get_image(self) -> np.ndarray:
        return self.pixels.astype(np.float32) / 255.0
    
    # Create an iterator that just renders the drawing
    def __iter__(self):
        while True:
            # Read events but don't consume them
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONUP:
                    self.pressed = False
                elif event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.MOUSEMOTION and self.pressed == True):
                    self.pressed = True
                    buttons = pygame.mouse.get_pressed()
                    # Draw when left mouse button is pressed
                    x, y = pygame.mouse.get_pos() # returns the position of mouse cursor
                    if buttons[0]:
                        self.draw(y, x, (255, 255, 255))
                    # Erase when right mouse button is pressed
                    elif buttons[2]:
                        self.draw(y, x, (0, 0, 0))
            self.render()
            yield events
            if pygame.display.get_init():
                pygame.display.flip()
            else:
                break


if __name__ == "__main__":
    game = DrawingGame(width = 800, height = 600, resolution = 28, blur = 0.4)

    for events in game:
        # Randomly pick a pixel and draw it
        # x = np.random.randint(0, width - 1)
        # y = np.random.randint(0, height - 1)
        # drawing.draw(y, x, (255, 255, 255))
        pass