from pathlib import Path

import imageio
import numpy as np

from sac import policy, utils
from sac.env import GymWrapper, env_ids
from sac.utils import get_paths, get_latest_run, load_json


env = GymWrapper('pendulum', monitor='./monitor')
hyp = load_json('experiments/pendulum.json')
actor = policy.make(env, hyp)

env.reset()
obs = env.reset().reshape(1, -1)
done = False
episode_reward = 0


paths = get_paths({
    'env-name': 'pendulum',
    'run-name': 'my-run'
})

checkpoint = Path('experiments/results/pendulum/my-run-2/checkpoints/test-episode-80')

actor.load_weights(checkpoint / 'actor.h5')

frames = []
while not done:
    _, _, action = actor(obs)
    frames.append(env.env.render('rgb_array'))
    next_obs, reward, done = env.step(np.array(action))
    episode_reward += reward
    obs = next_obs

print(episode_reward)

print('saving gif')
imageio.mimsave('./render.gif', frames, fps=44)
