# Deep Q-Learning

## Project Description
This project aims to train an agent using the **Deep Q-Learning** algorithm to reach a designated goal in a 4x4 environment. The agent must navigate while avoiding obstacles and collecting rewards along the way.

## Technologies Used
- **Python**
- **PyTorch**
- **NumPy**
- **Pygame**
- **Matplotlib**

## Game Rules
- The agent (blue) moves from the starting position to reach the goal.
- The goal (green square) is located at coordinates (3,3).
- Obstacles (red squares) cannot be passed by the agent.
- Stars (yellow stars) can be collected by the agent to gain extra rewards.
- The agent receives rewards or penalties based on its actions:
  - **Reaches the goal:** +1 reward
  - **Hits an obstacle:** -1 penalty
  - **Collects a star:** +0.6 reward
  - **Normal movement:** -0.03 reward

## Training Process
1. The agent performs actions either randomly or based on model predictions.
2. After each step, **state, action, reward, new state** are stored in memory.
3. Once enough data is collected, the **Replay Memory** is used to update the model.
4. As training progresses, the agent makes smarter decisions, avoiding obstacles and reaching the goal efficiently.
5. **At the end of training,** success rate, rewards collected, and step counts are analyzed using graphs.

## Future Improvements
- **Optimize learning speed** for larger environments.
- **Introduce more complex obstacles and reward structures.**
- **Use deeper neural networks** to enhance decision-making.
- **Experiment with alternative exploration methods** such as UCB or Boltzmann Exploration instead of epsilon-greedy.

This project serves as a fundamental application of **Deep Reinforcement Learning** and can be further improved in various ways.

