Can I start to split based on sampling v labelling v fitting?
- test & fill buffer should be one function
- what things read & save to disk

---

- json logging
- {"thread":"main","level":"INFO","loggerName":"mainLogger","message":{"foo":"bar"},"endOfBatch":false,"loggerFqcn":"org.apache.logging.log4j.spi.AbstractLogger","instant":{"epochSecond":1548434758,"nanoOfSecond":572000000},"threadId":1,"threadPriority":5}"

nem test / train data
- work started in tests/test_nem_dataset.py

from sac import make
sac.make

---

TODO
- test / train nem data episodes
- generate pretrained dataset energypylinear (use nem-data dataset as well)
- use dataset for pretraining
- tests for loading checkpoints

tool to analyze data  / logs
- import / export, reward

when are you ready to move over to energypy repo?
- data on s3
- thats about it :)

https://eprints.whiterose.ac.uk/159354/1/final_submitted_energy_storage_arbitrage_using_DRL%20%286%29.pdf
- their dataset
- dueling dqn

how i can improve
- transfer learning from off-policy learning of mlp data
- SAC - finer control, understand shape of the space
- different dataset

