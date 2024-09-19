import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from rlgym import RLgym

# Define the PPO network
class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()
        self.fc = nn.Linear(state_dim, 64)
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc(state))
        logits = self.policy(x)
        value = self.value(x)
        return logits, value

# PPO training loop
def train_ppo(env_name, num_episodes, lr=1e-3, gamma=0.99, epsilon=0.2, k_epochs=4):
    env = RLgym(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = PPOActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.item())

            # Calculate advantages
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            _, next_value = model(next_state_tensor)
            advantage = reward + (1 - done) * gamma * next_value - value

            # Compute policy loss and value loss
            old_log_prob = dist.log_prob(action)
            new_log_prob = dist.log_prob(action)
            ratio = torch.exp(new_log_prob - old_log_prob)
            policy_loss = -torch.min(ratio * advantage, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage).mean()
            value_loss = (reward + (1 - done) * gamma * next_value - value).pow(2).mean()

            optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            optimizer.step()

            state = next_state

# Example usage
train_ppo('CartPole-v1', num_episodes=1000)
