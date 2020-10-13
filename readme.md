# Soft Actor-Critic 

Reimplementation of the 2018 paper Soft Actor Critic - an off-policy, continuous actor-critic reinforcement learning algorithm.


## Setup

```bash
$ pip install -r requirements.txt
```

## Use

```bash
$ python3 main.py
```


## References

[Spinning Up notes on SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)

[Haarnoja et. al (2018) Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)


### Implementations

[Open AI Spinning Up - TF 1](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/sac/core.py)

[Stable Baselines - TF 1](https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/sac/policies.html)


### Environments

https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
