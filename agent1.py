import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from colorama import Fore
import math


random_state = 42
random.seed(random_state)
np.random.seed(random_state)
torch.manual_seed(random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)


pygame.init()

tile_size = 100
screen_size = 4 * tile_size
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Deep Q-Learning Agent")

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

player_pos = [0, 0]
finish_pos = [3, 3]
obstacles = [[0,3], [1, 1],[2, 3],[2, 1]]
stars = [[1, 3]]


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
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.01
batch_size = 16
memory = []

state_size = 16
action_size = 4
actions = ["up", "down", "left", "right"]

input_dim = 16
hidden_dim = 16
output_dim = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

def encode_state(player, finish, obstacles, stars):
    state = np.zeros((4, 4))
    state[player[0], player[1]] = 1
    state[finish[0], finish[1]] = 2
    for obs in obstacles:
        state[obs[0], obs[1]] = -1
    for star in stars:
        state[star[0], star[1]] = 0.5
    return state.flatten()

def move_player(direction, player_pos):
    row, col = player_pos
    if direction == "up" and row > 0:
        row -= 1
    elif direction == "down" and row < 3:
        row += 1
    elif direction == "left" and col > 0:
        col -= 1
    elif direction == "right" and col < 3:
        col += 1
    return [row, col]

def calculate_reward(player_pos):
    if player_pos == finish_pos:
        return 1
    elif player_pos in obstacles:
        return -1
    elif player_pos in stars:
        return 0.6
    else:
        return -0.03

def reset_stars():
    global stars
    stars = [[1, 3]]

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
    next_q_values = model(next_states).max(1)[0]
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    loss = loss_fn(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def plot_metrics(episodes, rewards, steps, win_rates, win_rate_past_100, stars_collected):
    plt.figure(figsize=(15, 10)) 

    plt.subplot(5, 1, 1)
    plt.plot(episodes, rewards, label="Total Reward", color="orange")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(episodes, steps, label="Steps", color="green")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.title("Steps per Episode")
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(episodes, win_rates, label="Win Rate", color="purple")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.title("Win Rate per Episode")
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(episodes, win_rate_past_100, label="Win Rate Past 100 Episodes", color="red")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate (Past 100 Episodes)")
    plt.title("Win Rate Past 100 Episodes")
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.bar(episodes, stars_collected, color="blue", label="Star Collected")
    plt.xlabel("Episodes")
    plt.ylabel("Star Collected")
    plt.title("Star Collected per Episode")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"metrics_plot.png")
    plt.close()

plt.figure(figsize=(15,7))
plt.savefig("metrics_plot.png")
plt.close()

for i in range(5, 0, -1):
    print(f"Eğitim {i} saniye içinde başlayacak...")
    time.sleep(1)

print("Eğitim başlıyor!")


rewards_per_episode = []
steps_per_episode = []
win_rates = []
episode_numbers = []
win_rate_past_100 = []
stars_collected_per_episode = []

episodes = 1000
win_count = 0

start_time = time.time()

for episode in range(episodes):
    player_pos = [0, 0]
    reset_stars()
    state = encode_state(player_pos, finish_pos, obstacles, stars)
    total_reward = 0
    steps = 0
    stars_collected = 0 
    done = False

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

        next_pos = move_player(actions[action], player_pos)
        reward = calculate_reward(next_pos)

        if next_pos in stars:
            stars.remove(next_pos)
            stars_collected += 1

        done = next_pos == finish_pos or next_pos in obstacles
        next_state = encode_state(next_pos, finish_pos, obstacles, stars)

        memory.append((state, action, reward, next_state, done))
        if len(memory) > 600:
            memory.pop(0)

        replay()

        state = next_state
        player_pos = next_pos
        total_reward += reward
        steps += 1

        draw_environment()
        pygame.display.flip()
        if episode + 1  > 950:
            pygame.time.delay(100)

    if player_pos == finish_pos:
        win_count += 1

    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)
    episode_numbers.append(episode + 1)
    win_rates.append(win_count / (episode + 1))
    stars_collected_per_episode.append(stars_collected) 

    if len(rewards_per_episode) > 100:
        win_rate_past_100.append(np.mean([1 if x > 0 else 0 for x in rewards_per_episode[-100:]]))
    else:
        win_rate_past_100.append(np.mean([1 if x > 0 else 0 for x in rewards_per_episode]))

    if (episode + 1)%100 == 0:
        print("Waiting for metrics report")
        plot_metrics(episode_numbers, rewards_per_episode, steps_per_episode, win_rates, win_rate_past_100, stars_collected_per_episode)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    episode_text = f"Episode {episode + 1}"
    total_reward_text = f"Total Reward: {total_reward:.4f}"
    steps_text = f"Steps: {steps}"
    win_rate_text = f"Win Rate: {win_rates[-1]:.2f}"
    win_rate_past_100_text = f"Win rate past 100 episodes: {win_rate_past_100[-1]:.2f}"
    stars_collected_text = f"Stars Collected: {stars_collected}"
    print(f"{Fore.CYAN}{episode_text}{Fore.RESET}: {Fore.YELLOW}{total_reward_text}{Fore.RESET}, {Fore.GREEN}{steps_text}{Fore.RESET}, {Fore.MAGENTA}{win_rate_text}{Fore.RESET}, {Fore.RED}{win_rate_past_100_text}{Fore.RESET}, {Fore.BLUE}{stars_collected_text}{Fore.RESET}")

print("Time left: ", time.time() - start_time)
pygame.quit()
sys.exit()