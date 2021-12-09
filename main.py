import sys, os
import datetime
curr_path = os.path.dirname(os.path.abspath(__file__)) # The absolute path of the current file
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # current time

import gym
import torch 
from plot import plot_rewards
from utils import save_results, make_dir
from ppo import Agent

class PPOConfig:

    def __init__(self) -> None:
        self.algo = "PPO" # the RL method name
        self.env_name = 'CartPole-v0' # the enviroment name
        self.continuous = False # continuous action or not 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_eps = 300 # trainning episodes
        self.eval_eps = 20 # testing episodes
        self.batch_size = 5
        self.gamma = 0.99
        self.n_epochs = 4
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.hidden_dim = 256
        self.update_fre = 20 # frequency of agent update

class PlotConfig:

    def __init__(self) -> None:
        self.algo = "PPO"
        self.env_name = 'CartPole-v0'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_path = curr_path + "/outputs/" + self.env_name + '/' + curr_time + '/results/'
        self.model_path = curr_path + '/outputs/' + self.env_name + '/' + curr_time + '/models/'
        self.save = True

def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env_name)
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # action_dim = env.action_space.shape[0]
    agent = Agent(state_dim, action_dim, cfg)
    return env, agent

def train(cfg, env, agent):
    print('......start trainning......')
    print(f'env: {cfg.env_name}, method: {cfg.algo}, device: {cfg.device}')
    
    rewards = [] # rewards of all episodes
    ma_rewards = [] # moving average rewards of all episodes
    steps = 0

    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            steps += 1
            ep_reward += reward
            agent.remember(state, action, prob, val, reward, done)
            if steps % cfg.update_fre == 0:
                agent.learn()
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1] + 0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1)%1 == 0:
            print(f'episode: {i_ep+1}/{cfg.train_eps}, rewards: {ep_reward:.2f}')
    print('......trainning end......')
    return rewards, ma_rewards

def eval(cfg, env, agent):
    print('......start testing......')
    print(f'env: {cfg.env_name}, method: {cfg.algo}, device: {cfg.device}')
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.eval_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            env.render()
            action, prob, val = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            ep_reward += reward
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1] + 0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f'episode: {i_ep+1}/{cfg.eval_eps}, rewards: {ep_reward:.2f}')
    print('......testing end......')
    env.close()
    return rewards, ma_rewards

if __name__ == '__main__':
    cfg = PPOConfig()
    plot_cfg = PlotConfig()
    # trainning
    env, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)
    agent.save_models(path=plot_cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)
    plot_rewards(rewards, ma_rewards, plot_cfg, tag='train')
    # testing
    env, agent = env_agent_config(cfg, seed=10)
    agent.load_models(path=plot_cfg.model_path)
    rewards,ma_rewards = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval',path=plot_cfg.result_path)
    plot_rewards(rewards, ma_rewards, plot_cfg,tag="eval")