## implementations

https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/sac/core.py

https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/sac/policies.html

## env

https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

## spinning up

https://spinningup.openai.com/en/latest/algorithms/sac.html

## notes

Learn three functions
- pi, Q1, Q2

Similar to TD3

- MSBE minimization (regress both Q func to shared target)
- shared target computed using target nets (two???) & clipped double Q trick
- target nets computed by polyak averaging the Q parameters

Dobule Q trick
- using minimum q value between Q1, Q2

Different from TD3

- entropy regularization
- next state actions come from current policy (not from target policy, not from experience!)
- target policy smoothing is not explicit - noise comes from stochastic policy

$a bar prime$ = next action sampled fresh

MSBE = mean square bellman error

Policy optimization
- max(V) = Q - alpha * log pi
- uses repamaretrization trick (take expectation over action -> expectation over noise)
- squashed (tanh) Gaussian policy
- parametrize both mean & stdev

Test time = use mean action (instead of sampling)


## entropy regularization 

Tradeoff between performance & policy randomness (for exploration)

Changes the optimal policy

- alpha = tradeoff coeff (controls exploration)
- can be fixed or 'entropy constrained'

Changes the value func & Q func

- V includes entropy bonus from all timesteps
- Q func includes entropy bonus from all timesteps except first (but this can be diffeerent)

Can define the value func in terms of Q & H

Can define the Q func in terms of V & reward
