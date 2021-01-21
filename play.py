import imageio
import numpy as np

from env import GymWrapper, env_ids
from policy import make_policy


env = GymWrapper(env_ids['lunar'], monitor='./monitor')
actor = make_policy(env, size_scale=6)

env.reset()
obs = env.reset().reshape(1, -1)
done = False
episode_reward = 0

actor.load_weights('./old-tensorboards/run5-success/pol.h5')

frames = []
while not done:
    _, _, action = actor(obs)
    frames.append(env.env.render('rgb_array'))
    next_obs, reward, done = env.step(np.array(action))
    episode_reward += reward
    obs = next_obs

print(episode_reward)

imageio.mimsave('./render.gif', frames, fps=44)
