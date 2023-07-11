import pygame


def render_multiline(lines: list[str], font: pygame.font.Font, color: tuple[int, int, int], background_color: tuple[int, int, int],
                     x: int, y: int, screen: pygame.Surface):
    for i, line in enumerate(lines):
        text = font.render(line, True, color, background_color)
        screen.blit(text, (x, y + i * font.get_height()))