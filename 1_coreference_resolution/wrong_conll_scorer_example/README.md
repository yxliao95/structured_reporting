
# How to trigger?

Modify the token "1)" at line 2 column 4, and run the following command. You will see the error.

```
cd path/to/fast-coref
mkdir coref_resources; cd coref_resources/
git clone https://github.com/conll/reference-coreference-scorers
cd ../..
fast-coref/coref_resources/reference-coreference-scorers/scorer.pl muc wrong_conll_scorer_example/test_gt.conll wrong_conll_scorer_example/test_predict.conll none
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

# How we address?

We replaced the () by [] (regex: `\(?[^A-Za-z]+\)?`). See i2b2_raw2conll.Token.get_conll_str() for details.
