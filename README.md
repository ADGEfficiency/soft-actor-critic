# Soft Actor-Critic 

Reimplementation of the 2018 paper Soft Actor Critic - an off-policy, continuous actor-critic reinforcement learning algorithm.

Currently the implementation has been tested on the Pendulum-v0 gym environment:

![](assets/pendulum.png)


## Setup

```bash
$ brew install swig
$ pip install -r requirements.txt
```

## Use

```bash
$ python3 main.py
```


## References

[Spinning Up notes on SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)

[Haarnoja et. al (2018) Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) - [pdf](https://arxiv.org/pdf/1801.01290.pdf)


### Implementations

[Open AI Spinning Up - TF 1](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/sac/core.py)

https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/sac/sac.py

[Stable Baselines - TF 1](https://stable-baselines.readthedocs.io/en/master/modules/sac.html)

https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/sac/sac.html#SAC

SLM-Lab
- [sac.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/sac.py)
- [lunar lander benchmark hyperparameters](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/sac/sac_lunar.json)


### Environments

https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
