import gym
import model_A2C as ma2c
from a2c_agent import A2CAgent
import os
import tensorflow as tf

env = gym.make('gym_go:go-v0', size=19, komi=0)

env.seed(42)

# Game loop
#model = ma2c.ActorCritic(env.action_space.n)
model = None
obs = env.reset()

agent = A2CAgent(model)
reward_sum = agent.train(env)
print(reward_sum)
test_sum = agent.test(env)
print(test_sum)