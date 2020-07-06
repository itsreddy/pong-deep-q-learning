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

def load_existing_model(model_path, policy_net, target_net):
    policy_net.load_state_dict(torch.load(model_path))
    target_net.load_state_dict(policy_net.state_dict())
    return policy_net, target_net


# main

USE_CUDA = torch.cuda.is_available()
base_path = os.getcwd()

create_save_folder(base_path)
env = init_env()

num_frames = 2500000
batch_size = 64
gamma = 0.99
    
replay_initial = 10000
replay_buffer = ReplayBuffer(100000)

policy_net = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
target_net = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)

mse = torch.nn.MSELoss()

if USE_CUDA:
    policy_net = policy_net.cuda()
    target_net = target_net.cuda()
    mse = mse.cuda()

optimizer = optim.Adam(policy_net.parameters(), lr=0.00001)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start \
                - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


losses = []
all_rewards = []
episode_reward = 0

TARGET_UPDATE = 10
EPISODE_COUNT = 0

state = env.reset()
losses_mean = []

for frame_idx in range(2090000+1, 2090000+num_frames + 1):

    epsilon = epsilon_by_frame(frame_idx)
    action = target_net.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        if EPISODE_COUNT % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        state = env.reset()
        all_rewards.append(episode_reward)
        EPISODE_COUNT += 1
        print("episodes done: ", EPISODE_COUNT, "episode reward: ", episode_reward)
        episode_reward = 0
        if len(replay_buffer) > replay_initial:
            lossmean = np.mean(losses)
            print('#Frame: %d, mean loss: %f' % (frame_idx, lossmean))
            losses_mean.append(lossmean)
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
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print("Last-10 average reward:", np.mean(all_rewards[-10:]), " lr: ", optimizer.param_groups[0]['lr'])
    
    if frame_idx > 500000 and frame_idx % 5000 == 0 and len(replay_buffer) > replay_initial:
        print("saving model")
        torch.save(policy_net.state_dict(), models_path + "/model_" + "%d.pth" % (frame_idx))

