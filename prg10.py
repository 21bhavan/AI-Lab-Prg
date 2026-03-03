import matplotlib
matplotlib.use('TkAgg')   # Fix backend issue
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Grid & Hyperparameters
GRID_SIZE = 5
GAMMA = 0.9
ALPHA = 0.1
EPSILON = 0.2
EPISODES = 500

ACTIONS = ['up', 'down', 'left', 'right']
ACTION_MOVES = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

START = (0, 0)
GOAL = (4, 4)

# Initialize Q-table
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Reward grid
REWARD_GRID = np.full((GRID_SIZE, GRID_SIZE), -1)
REWARD_GRID[GOAL] = 10


def move(state, action):
    """Returns next state and reward."""
    move = ACTION_MOVES[action]
    next_state = (state[0] + move[0], state[1] + move[1])

    # Check wall condition
    if not (0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE):
        return state, -10  # penalty for hitting wall
    else:
        return next_state, REWARD_GRID[next_state]


def choose_action(state):
    """Epsilon-greedy action selection."""
    if np.random.rand() < EPSILON:
        return np.random.choice(ACTIONS)  # explore
    else:
        return ACTIONS[np.argmax(Q_table[state])]  # exploit


def train():
    """Q-learning training loop."""
    rewards = []

    for episode in range(EPISODES):
        state = START
        total_reward = 0

        while state != GOAL:
            action = choose_action(state)
            next_state, reward = move(state, action)

            action_index = ACTIONS.index(action)

            # Q-learning update formula
            Q_table[state][action_index] += ALPHA * (
                reward + GAMMA * np.max(Q_table[next_state])
                - Q_table[state][action_index]
            )

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    return rewards


# Train agent
rewards = train()

# Plot reward performance
plt.plot(rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q-learning Performance")
plt.show()

# Show learned Q-values heatmap
sns.heatmap(np.max(Q_table, axis=2), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Q-values Heatmap")
plt.show()