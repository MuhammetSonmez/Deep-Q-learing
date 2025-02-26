import pygame
import sys
import random
import math

pygame.init()

tile_size = 100
screen_size = 3 * tile_size
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Snake Game")

WHITE = (255, 255, 255)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (0, 0, 128)
YELLOW = (255, 223, 0)
RED = (255, 0, 0)

snake = [[0, 0]]
stars = [[1, 2]]

direction = None

collected_stars = 0

def draw_food(center, size):
    pygame.draw.circle(screen, YELLOW, center, size)

def draw_environment():
    screen.fill(LIGHT_BLUE)
    for index, segment in enumerate(snake):
        segment_center = (segment[1] * tile_size + tile_size // 2, segment[0] * tile_size + tile_size // 2)
        if index == 0:
            pygame.draw.circle(screen, RED, segment_center, tile_size // 3)
        else:
            pygame.draw.circle(screen, DARK_BLUE, segment_center, tile_size // 4)
    for star in stars:
        star_center = (star[1] * tile_size + tile_size // 2, star[0] * tile_size + tile_size // 2)
        draw_food(star_center, tile_size // 4)

def move_snake():
    global snake, stars, direction, collected_stars

    if direction is None:
        return

    head = snake[0]
    if direction == "up":
        new_head = [head[0] - 1, head[1]]
    elif direction == "down":
        new_head = [head[0] + 1, head[1]]
    elif direction == "left":
        new_head = [head[0], head[1] - 1]
    elif direction == "right":
        new_head = [head[0], head[1] + 1]

    new_head[0] %= 3
    new_head[1] %= 3

    if new_head in snake:
        print("Game Over! Yılan kendine çarptı.")
        pygame.quit()
        sys.exit()

    snake.insert(0, new_head)

    if new_head in stars:
        stars.remove(new_head)
        collected_stars += 1
        if collected_stars == 8:
            print("Tebrikler! 8 yıldız topladınız, oyunu kazandınız!")
            pygame.quit()
            sys.exit()
    else:
        snake.pop()

    if not stars:
        while True:
            new_star = [random.randint(0, 2), random.randint(0, 2)]
            if new_star not in snake:
                stars.append(new_star)
                break

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != "down":
                direction = "up"
                move_snake()
            elif event.key == pygame.K_DOWN and direction != "up":
                direction = "down"
                move_snake()
            elif event.key == pygame.K_LEFT and direction != "right":
                direction = "left"
                move_snake()
            elif event.key == pygame.K_RIGHT and direction != "left":
                direction = "right"
                move_snake()

    draw_environment()
    pygame.display.flip()

pygame.quit()
sys.exit()
