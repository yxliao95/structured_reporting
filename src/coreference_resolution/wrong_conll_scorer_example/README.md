
# How to trigger?

1. Modify any token to "1)" (at column 4). Run the following command, and you will see the error.
2. Assign an singleton to either file (at the end of the line, press TAB and then input "(2)" or any id). Run the following command, and you will see the error from the MUC metric.
   1. It means that the MUC metric does not consider singletons.

```
cd path/to/src/coreference_resolution/wrong_conll_scorer_example
git clone https://github.com/conll/reference-coreference-scorers
reference-coreference-scorers/scorer.pl muc err_gt.conll err_predict.conll none
```

# Error outputs

- Example 1: using `(1)`.

```
====== TOTALS =======
Identification of Mentions: Recall: (1 / 1) 100%        Precision: (1 / 2) 50%  F1: 66.66%      <--------- Should be Precision: (1 / 1)
--------------------------------------------------------------------------
Coreference: Recall: (0 / 0) 0% Precision: (0 / 0) 0%   F1: 0%
--------------------------------------------------------------------------
```

- Example 2: using `1)`.

```
Detected the end of a mention [11](0) without begin (?,0) at /home/yuxiangliao/PhD/workspace/git_clone_repos/fast-coref/coref_resources/reference-coreference-scorers/lib/CorScorer.pm line 292, <F> line 24.
```

- Example 3: Given two mentions with cluster id `(2)` in err_gt.conll, but only assign one `{2}` in err_predict.conll.

```
METRIC muc:

====== TOTALS =======
Identification of Mentions: Recall: (3 / 4) 75% Precision: (3 / 3) 100% F1: 85.71%
--------------------------------------------------------------------------
Coreference: Recall: (1 / 2) 50%        Precision: (1 / 1) 100% F1: 66.66%           <--------- Should be Precision: (1 / 2)
--------------------------------------------------------------------------
```

# How we address?

We replaced the () by [] (regex: `\(?[^A-Za-z]+\)?`). See i2b2_raw2conll.Token.get_conll_str() for details.
