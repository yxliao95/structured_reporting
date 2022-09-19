## MIMIC-CXR Output

```JSON
{
  "Using": {
    "Library": "spaCy",
    "Model": "en_core_web_md",
    "Pipeline enable": "['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer']",
    "Pipeline disable": "['ner']"
  },
  "Number of input records": 227835,
  "Number of not empty records": {
    "findings": 156011,
    "impression": 189465,
    "provisional_findings_impression": 200,
    "findings_and_impression": 10589
  },
  "Time cost": "15214.51"
}

{
  "Using": {
    "Library": "CoreNLP",
    "Properties": {
      "annotators": "tokenize, ssplit, pos, lemma, ner, depparse, coref",
      "coref_algorithm": "statistical"
    },
    "Output": "All annotators' results"
  },
  "Number of input records": 227835,
  "Number of not empty records": {
    "findings": 156011,
    "impression": 189465,
    "provisional_findings_impression": 200,
    "findings_and_impression": 10589
  },
  "Time cost": "7542.94"
}

{
  "Using": {
    "Library": "CoreNLP",
    "Properties": {
      "annotators": "tokenize, ssplit, pos, lemma, ner, parse, coref",
      "coref_algorithm": "neural"
    },
    "Output": "Only the last coref annotator's results"
  },
  "Number of input records": 227835,
  "Number of not empty records": {
    "findings": 156011,
    "impression": 189465,
    "provisional_findings_impression": 200,
    "findings_and_impression": 10589
  },
  "Time cost": "25691.89"
}

{
  "Using": {
    "Library": "CoreNLP",
    "Properties": {
      "annotators": "tokenize, ssplit, pos, lemma, ner, parse, dcoref"
    },
    "Output": "Only the last coref annotator's results"
  },
  "Number of input records": 227835,
  "Number of not empty records": {
    "findings": 156011,
    "impression": 189465,
    "provisional_findings_impression": 200,
    "findings_and_impression": 10589
  },
  "Time cost": "41727.53"
}
```

## i2b2 output

When using CoreNLP-dcoref, some of the sentences somehow failed in getting the dependencies.

For example, clinical-23 will trigger an error that fails to perform dependency parsing on some of the sentences (seemly because the sentence is too long). In that case, the corresponding value in the CSV file is set to "-1" (str)