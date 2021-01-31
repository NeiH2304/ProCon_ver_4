from __future__ import division
import random
from copy import deepcopy as dcopy

from src.environment import Environment
from src.agents import Agent
from read_input import Data
from GameBoard.game_board import BoardGame
import pygame
from itertools import count
import torch
import numpy as np
from src.utils import flatten, vizualize
from collections import deque
import time
import gym

env = gym.make('BipedalWalker-v3')
# env = gym.make('Pendulum-v0')

MAX_EPISODES = 5000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)


def test_env(opt): 
    input_dim_actor = S_DIM
    input_dim_critic = S_DIM
    max_agents = 8
    max_actions = A_DIM
    num_agents = 1
    
    trainer = Agent(opt.gamma, opt.lr_actor, opt.lr_critic, input_dim_actor, input_dim_critic, num_agents, max_agents, max_actions,
                    opt.replay_memory_size, opt.batch_size, 'agent_procon_1', opt.load_checkpoint,
                    opt.saved_path, Environment())
    
    for _ep in range(opt.n_epochs):
        print('Training_epochs: {}'.format(_ep + 1))
        observation = env.reset()
        
        done = False
        
        for _iter in range(MAX_STEPS): 
            env.render()
            state = np.float32(observation)
            
            action = trainer.get_exploration_action(state)
    		# if _ep%5 == 0:
    		# 	# validate every 5th episode
    		# 	action = trainer.get_exploitation_action(state)
    		# else:
    		# 	# get action based on observation, use exploration policy here
    		# 	action = trainer.get_exploration_action(state)

            new_observation, reward, done, info = env.step(action)

    		# # dont update if this is validation
    		# if _ep%50 == 0 or _ep>450:
    		# 	continue

            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation)
                # push this exp in ram
                trainer.memories.store_transition(state, action, reward, new_state)

            observation = new_observation

    		# perform optimization
            trainer.optimize()
                
            if done:
                break
            
        print('Completed episodes')

