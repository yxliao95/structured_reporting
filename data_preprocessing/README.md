# Process the i2b2 2011 - Coreference task dataset

Check the config in `/config/data_preprocessing.yaml`.

Download and copy `manually_processed_records.json` to `/data_preprocessing/resources` if you want to get the ultimately preprocessed data.

Modify `/config/data_preprocessing/db.yaml` if you want mysql output.

Run:

```bash
python resolve_raw_text.py
```
