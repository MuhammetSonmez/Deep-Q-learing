import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

pygame.init()
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

time_flag = False

tile_size = 100
screen_size = 3 * tile_size
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Snake Game with DQN")

WHITE = (255, 255, 255)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (0, 0, 128)
YELLOW = (255, 223, 0)
RED = (255, 0, 0)

predefined_stars = [
                    [0, 1],
                    [2, 2],     
                    [0, 2],
                    [1, 1],
                    [2, 0],
                    [1, 0]]

star_index = 0

def generate_new_star():
    global star_index
    if star_index < len(predefined_stars):
        new_star = predefined_stars[star_index]
        star_index += 1
        return [new_star]
    return []

snake = [[0, 0]]
stars = generate_new_star()
direction = "right"
collected_stars = 0

actions = ["up", "down", "left", "right"]

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999
learning_rate = 0.05
batch_size = 32
memory = deque(maxlen=200)

state_size = 9
action_size = 4
input_dim = state_size
hidden_dim = 16
output_dim = action_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN(input_dim, hidden_dim, output_dim).to(device)
target_model = DQN(input_dim, hidden_dim, output_dim).to(device)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

def encode_state(snake, stars):
    state = np.zeros((3, 3))
    for segment in snake:
        if 0 <= segment[0] < 3 and 0 <= segment[1] < 3:
            state[segment[0], segment[1]] = 1
    for star in stars:
        if 0 <= star[0] < 3 and 0 <= star[1] < 3:
            state[star[0], star[1]] = 0.5
    return state.flatten()

def is_opposite_direction(current_direction, new_direction):
    opposites = {
        "up": "down", "down": "up",
        "left": "right", "right": "left"
    }
    return opposites.get(current_direction) == new_direction

def manhattan_distance(snake_head, star):
    return abs(snake_head[0] - star[0]) + abs(snake_head[1] - star[1])

def move_snake(direction, snake):
    global stars
    if time_flag:
        pygame.time.delay(150)
        
    head = snake[0]
    new_head = head[:]
    if not is_opposite_direction(direction, new_direction):
        if direction == "up":
            new_head[0] -= 1
        elif direction == "down":
            new_head[0] += 1
        elif direction == "left":
            new_head[1] -= 1
        elif direction == "right":
            new_head[1] += 1

    if head[0] == 0 and direction == "up":
            return 0
    if head[0] == 2 and direction == "down":
        return 0
    if head[1] == 0 and direction == "left":
        return 0
    if head[1] == 2 and direction == "right":
        return 0
    if new_head in snake:
        return None

    snake.insert(0, new_head)

    if new_head in stars:
        stars.remove(new_head)
        return 10
    
    min_distance = float('inf')
    for star in stars:
        min_distance = min(min_distance, manhattan_distance(new_head, star))

    reward = -0.01
    if min_distance == 1:
        reward += 0.4
    elif min_distance == 2:
        reward += 0.2

    snake.pop()

    return reward

def replay():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_model(next_states).max(1)[0]
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    loss = loss_fn(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
        pygame.draw.circle(screen, YELLOW, star_center, tile_size // 4)

episodes = 1000
rewards_per_episode = []

for episode in range(episodes):
    if episode > episodes - (episodes * 0.2):
        time_flag = True

    snake = [[0, 0]]
    star_index = 0
    stars = generate_new_star()
    direction = "right"
    total_reward = 0
    done = False
    state = encode_state(snake, stars)
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if random.random() < epsilon:
            action = random.choice(range(action_size))
        else:
            with torch.no_grad():
                q_values = model(torch.tensor(state, dtype=torch.float32).to(device))
                action = torch.argmax(q_values).item()

        new_direction = actions[action]
        if not is_opposite_direction(direction, new_direction):
            direction = new_direction

        reward = move_snake(direction, snake)
        if reward is None:
            done = True
            reward = -10

        next_state = encode_state(snake, stars)
        memory.append((state, action, reward, next_state, done))

        replay()

        state = next_state
        total_reward += reward

        if len(stars) == 0:
            stars = generate_new_star()

        draw_environment()
        pygame.display.flip()

    rewards_per_episode.append(total_reward)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())

pygame.quit()
