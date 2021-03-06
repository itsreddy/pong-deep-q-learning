import math, random, os, time
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from model import QLearner
from loss import compute_td_loss
from replay_buffer import ReplayBuffer

def create_save_folder(base_path):
    timeasname = time.asctime(time.localtime(time.time())).replace(" ", "-").replace(":", "-")
    results_path = base_path + "outs/{}/".format(timeasname)
    models_path = results_path +  "models/"
    os.makedirs(models_path, exist_ok=True)

def init_env(env_id="PongNoFrameskip-v4")
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    return env

def init_model(env):
    policy_net = QLearner(env)
    target_net = QLearner(env)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    return policy_net, target_net

def load_existing_model(model_path, policy_net, target_net):
    policy_net.load_state_dict(torch.load(model_path))
    target_net.load_state_dict(policy_net.state_dict())
    return policy_net, target_net


class Options:
    def __init__():
        self.cuda = torch.cuda.is_available()
        self.base_path = os.getcwd()
        self.num_frames = 1000000 # probably need more
        self.batch_size = 64
        self.lr = 0.00001
        self.gamma = 0.99
        self.replay_initial = 10000
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 30000
        self.target_update = 5
        self.episode_reward_print_interval = 1
        self.avg_reward_print_interval = 10
        self.model_save_interval = 5000
        self.model_save_init = 500000
        self.frame_idx_start = None
        self.model_path = None


# main

opt = Options()

create_save_folder(opt.base_path)
env = init_env()
replay_buffer = ReplayBuffer(100000)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)

policy_net, target_net = init_model(env)
mse = torch.nn.MSELoss()

if opt.cuda:
    policy_net = policy_net.cuda()
    target_net = target_net.cuda()
    mse = mse.cuda()

optimizer = optim.Adam(policy_net.parameters(), lr=opt.lr)

epsilon_start = opt.epsilon_start
epsilon_final = opt.epsilon_final
epsilon_decay = opt.epsilon_decay
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start \
                - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


episode_reward = 0
episode_count = 0

losses = []
episode_rewards = []
episode_mean_loss = []

if opt.frame_idx_start is None:
    frame_idx_start = 0
else:
    policy_net, target_net = load_existing_model(opt.model_path)

state = env.reset()

for frame_idx in range(1, frame_idx_start + num_frames + 1):

    epsilon = epsilon_by_frame(frame_idx)
    action = target_net.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        if episode_count % opt.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        state = env.reset()

        episode_rewards.append(episode_reward)
        episode_count += 1
        if episode_count % opt.episode_reward_print_interval:
            print("episodes done: ", episode_count, 
                            "episode reward: ", episode_reward)

        episode_reward = 0

        if len(replay_buffer) > replay_initial:
            mean_loss = np.mean(losses)
            if episode_count % opt.episode_reward_print_interval:
                print('# Frame: %d, mean loss: %f' % (frame_idx, lossmean))
            episode_mean_loss.append(mean_loss)
            losses = []
            print('-'*20)

    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(batch_size, gamma, mse, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.cpu().numpy())
        # losses.append(loss.data.cpu().numpy())
        # scheduler.step(np.mean(losses))

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('Frame: %d, preparing replay buffer' % frame_idx)

    # if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:

    if episode_count % opt.avg_reward_print_interval and\
                                 len(replay_buffer) > replay_initial:
        print("Last-10 episode average reward:", np.mean(episode_rewards[-10:]))
    

    if frame_idx > opt.model_save_init and frame_idx % opt.model_save_interval\
                                 == 0 and len(replay_buffer) > replay_initial:
        print("saving model at frame#: ", frame_idx)
        torch.save(policy_net.state_dict(), models_path +\
                                         "/model_" + "%d.pth" % (frame_idx))