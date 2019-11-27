
Sane words challenge
======================

Guess if a given word is a correct Polish word in a given domain. Additionally, you have the information on reported frequency of the word in source texts.

Each entry in training data set is of the form: __Sane (0 or 1), Domain, Word, Frequency__.
Evaluation metric is F2-score.


Directory structure
-------------------

* `README.md` — this file
* `config.txt` — configuration file
* `train/` — directory with training data
* `train/train.tsv` — train set
* `dev-0/` — directory with dev (test) data
* `dev-0/in.tsv` — input data for the dev set
* `dev-0/expected.tsv` — expected (reference) data for the dev set
* `test-A` — directory with test data
* `test-A/in.tsv` — input data for the test set
* `test-A/expected.tsv` — expected (reference) data for the test set
