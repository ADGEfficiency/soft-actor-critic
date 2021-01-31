restart from checkpoint
 
random seeds

Can I start to split based on sampling v labelling v fitting?
- episode, test & fill buffer random all require running episodes
- train dosen't
- test & fill buffer should be one function
- what things read & save to disk

What happen when you use a run name twice?
- if same run_name via cli, it will delete
- other mode is restart!

Do average of test train rewards separately
- maybe keep separate lists (plus one combined)

ideas
- make pol, make qfunc (not initialize)
- hyperparameters should be loaded in (experiment class?)
- some things are added to hyperparameters...
- where does buffer live?
- optimizers should be dict

runner
- counters can hold rewards as well - runner class?
- writer
- counter
- hyperparameters

wanted func
- start experiment from file
- restart experiment from checkpoint (based on tests)

blog post

- explain the a tilde ' = next state action of current pol, a tidle = current state
policy entropy

features
- test episodes
- chekcpoint & restarts
