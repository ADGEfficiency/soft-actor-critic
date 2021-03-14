Can I start to split based on sampling v labelling v fitting?
- episode, test & fill buffer random all require running episodes
- train dosen't
- test & fill buffer should be one function
- what things read & save to disk

energypy build
- energy balances in-line in step

$ make monitor

14 March

- abstract out business logic of battery storage, so it can be used easily in many battery
- test / train nem data episodes
- generate pretrained dataset energypylinear (use nem-data dataset as well)
- use dataset for pretraining
- use dataset for comparing

tool to analyze data  / logs
- import / export, reward

reward clipping

https://eprints.whiterose.ac.uk/159354/1/final_submitted_energy_storage_arbitrage_using_DRL%20%286%29.pdf
- their dataset
- dueling dqn

how i can improve
- transfer learning from off-policy learning of mlp data
- SAC - finer control, understand shape of the space
- different dataset

when are you ready to move?
- data on s3
- thats about it :)

tests for loading checkpoints
