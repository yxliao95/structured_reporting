# The scorer of CoNLL-2011/2012 shared tasks on coreference resolution

## Scoring logic

All the `docs` in conll files will be used for calculation. In gt.conll and pred.conll, the `doc` with same `docId` are matched. If a `doc` exists only in one of the conll file, it would affect either the recall or precision (e.g. a missing `doc` in pred.conll means its corefs are all missing)

The acutal `token string`, `token id`, `sentence segmentation` (new line) do not affect the measurement. The `empty line`, including its `number` and `position`, do not affect the result.

However, the position of the coref label (i.e. at which row) is the key and should be identical, otherwise the results would be wrong. In other words, the socrer will align every lines (e.g. gt row1 <-> pred row1, gt row_n <-> pred row_n). Then only the last column (`coref labels`) will be used for scoring. The `value of the coref label` does not affect the mention-level assesment and the coref-level (group-level) assesment (e.g. gt use value `0` and pred use value `1`).

## How to trigger?

1. Modify any token to "1)" (at column 4). Run the following command, and you will see the error.
2. Assign an singleton to either file (at the end of the line, press TAB and then input "(2)" or any id). Run the following command, and you will see the error from the MUC metric.
   1. It means that the MUC metric does not consider singletons.

```
cd path/to/src/coreference_resolution/wrong_conll_scorer_example
git clone https://github.com/conll/reference-coreference-scorers
reference-coreference-scorers/scorer.pl muc err_gt.conll err_predict.conll none
```

## Error outputs

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

## How we address?

We replaced the () by [] (regex: `\(?[^A-Za-z]+\)?`). See i2b2_raw2conll.Token.get_conll_str() for details.
