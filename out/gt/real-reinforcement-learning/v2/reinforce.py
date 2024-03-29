import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.transforms as T
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=int, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 1)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()
env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)
model = Policy()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(Variable(state))
    action = probs.multinomial()
    model.saved_actions.append(action)
    return action.data

def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / rewards.std()
    for (action, r) in zip(model.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(model.saved_actions, [None for _ in model.saved_actions])
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]
running_reward = 10
for i_episode in count(1):
    state = env.reset()
    for t in range(10000):
        action = select_action(state)
        (state, reward, done, _) = env.step(action[0, 0])
        if args.render:
            env.render()
        model.rewards.append(reward)
        print('LOG STMT: Action = %s, reward = %s' % (action, reward))
        if done:
            break
    running_reward = running_reward * 0.99 + t * 0.01
    finish_episode()
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i_episode, t, running_reward))
    if running_reward > 200:
        print('Solved! Running reward is now {} and the last episode runs to {} time steps!'.format(running_reward, t))
        break
