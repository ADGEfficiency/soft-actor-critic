

from policy import make_policy

from env import GymWrapper, env_ids

env = GymWrapper(env_ids['lunar'])
actor = make_policy(env, size_scale=6)

env.reset()
import numpy as np
obs = env.reset().reshape(1, -1)
done = False
episode_reward = 0

actor.load_weights('pol.h5')

while not done:
    _, _, action = actor(obs)
    env.env.render()
    next_obs, reward, done = env.step(np.array(action))
    episode_reward += reward
