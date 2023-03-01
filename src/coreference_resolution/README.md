# Process the i2b2 2011 - Coreference task dataset

Please register and download the 2011 - Coreference (Clinical) dataset from [here](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).

## Usage

### Create BRAT data for annotation

Follow the instructions in ``create_brat_ann_data.ipynb``.

## Create training data for fast-coref

1. Put the annotated data to ``../output/brat_annotation``.
2. Follow the instruction in ``resolve_brat_ann_data.ipynb``.
   - It would output csv files with resolved conll labels
3. Follow the instruction in `convert_conll&jsonlines.ipynb`
   - It would output a `conll` dir and a `longformer` dir for fast-coref model training