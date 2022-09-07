# Usage

If the scripts fail to import modules, please make sure the following paths are added to the PYTHONPATH environment variable.

```bash
# /path/to/fast-coref/src
export PYTHONPATH=/home/yuxiangliao/PhD/workspace/git_clone_repos/fast-coref/src
# /path/to/strcuture_reporting/src
export PYTHONPATH=src/:$PYTHONPATH
```

## Cautions

The column of CSV files might not follow the same order. When the reports are being processed by CoreNLP with multiple coref annotators, some of the reports may not be successfully processed in the first round. We will re-run the coref annotators on `unfinished records` in the second round. This will lead to a different order of the columns for those second-round-processed reports. For those disorder reports' sid, you can find them from `/output/nlp_ensamble/run.log or corenlp_unfinished_records.log`