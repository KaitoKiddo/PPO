import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim 
from torch.distributions.categorical import Categorical

class PPOMemory:

    def __init__(self, batch_size) -> None:
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        val = self.critic(state)

        return val

class Agent:
    def __init__(self, state_dim, action_dim, cfg) -> None:
        self.gamma = cfg.gamma
        self.continuous = cfg.continuous
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device
        self.actor = ActorNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, cfg.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PPOMemory(cfg.batch_size)
        self.loss = 0

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, path):
        print('......saving models......')
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
        T.save(self.actor.state_dict(), actor_checkpoint)
        T.save(self.critic.state_dict(), critic_checkpoint)

    def load_models(self, path):
        print('......loading models......')
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(path, 'ppo_critic.pt')
        self.actor.load_state_dict(T.load(actor_checkpoint)) 
        self.critic.load_state_dict(T.load(critic_checkpoint)) 

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        # action = T.tensor([action], dtype=T.float).to(self.device)

        probs = T.squeeze(dist.log_prob(action)).item()
        if self.continuous:
            action = T.tanh(action)
        else:
            action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            # compute advantage
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.device)
            # SGD
            values = T.tensor(values).to(self.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch]).to(self.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.loss = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        self.memory.clear_memory()