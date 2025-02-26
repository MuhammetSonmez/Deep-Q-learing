import pygame
import sys
import random
import math

pygame.init()

tile_size = 100
screen_size = 4 * tile_size
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Pygame Environment")

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

environment = [["" for _ in range(4)] for _ in range(4)]
player_pos = [0, 0]
finish_pos = [3, 3]
obstacles = []
stars = [[1, 3]]

exit_flag = False

def setup(random_map=True):
    global environment, obstacles
    environment[0][0] = "player"
    environment[finish_pos[0]][finish_pos[1]] = "finish"

    if random_map:
        num_obstacles = random.randint(2, 4)
        obstacles = []

        while len(obstacles) < num_obstacles:
            row = random.randint(0, 3)
            col = random.randint(0, 3)

            if [row, col] not in obstacles and [row, col] != player_pos and [row, col] != finish_pos and [row, col] not in stars:
                if not any(abs(row - r) <= 1 and abs(col - c) <= 1 for r, c in obstacles):
                    obstacles.append([row, col])
    else:
        obstacles = [[0, 3], [1, 1], [2, 3], [2, 1]]

    for obstacle in obstacles:
        environment[obstacle[0]][obstacle[1]] = "obstacle"


def draw_star(center, size):
    points = []
    for i in range(5):
        angle = i * 144
        x = center[0] + size * math.cos(math.radians(angle))
        y = center[1] + size * math.sin(math.radians(angle))
        points.append((x, y))
    pygame.draw.polygon(screen, YELLOW, points)

def draw_environment():
    screen.fill(WHITE)
    for row in range(4):
        for col in range(4):
            rect = pygame.Rect(col * tile_size, row * tile_size, tile_size, tile_size)
            if [row, col] == player_pos:
                pygame.draw.circle(screen, BLUE, (col * tile_size + tile_size // 2, row * tile_size + tile_size // 2), tile_size // 4)
            elif [row, col] == finish_pos:
                pygame.draw.rect(screen, GREEN, rect)
            elif [row, col] in obstacles:
                pygame.draw.rect(screen, RED, rect)
            elif [row, col] in stars:
                star_center = (col * tile_size + tile_size // 2, row * tile_size + tile_size // 2)
                draw_star(star_center, tile_size // 4)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)


def move_player(direction):
    global player_pos, exit_flag, stars
    row, col = player_pos

    if direction == "up" and row > 0:
        row -= 1
    elif direction == "down" and row < 3:
        row += 1
    elif direction == "left" and col > 0:
        col -= 1
    elif direction == "right" and col < 3:
        col += 1

    if [row, col] in obstacles:
        print("Engelin üzerindesiniz!")
        exit_flag = True

    player_pos = [row, col]

    if player_pos in stars:
        stars.remove(player_pos)

    if exit_flag:
        exit()

map_choice = "static"
if map_choice == "static":
    setup(random_map=False)
elif map_choice == "random":
    setup(random_map=True)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                move_player("up")
            elif event.key == pygame.K_DOWN:
                move_player("down")
            elif event.key == pygame.K_LEFT:
                move_player("left")
            elif event.key == pygame.K_RIGHT:
                move_player("right")

    draw_environment()
    pygame.display.flip()

    if player_pos == finish_pos:
        print("Tebrikler! Hedefe ulaştınız!")
        running = False

pygame.quit()
sys.exit()
