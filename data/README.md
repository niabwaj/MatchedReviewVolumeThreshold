# Data folder

The data are clinical records from Internetpsykiatrien, Centre for Digital Psychiatry, Region of Southern Denmark, held under Danish data protection law, and this repository holds code only. The access route is in the Data Availability statement of the manuscript.

With access granted, the pipeline expects the files below. `data_clean.ipynb` writes the two cleaned files, which go into `raw_data/`. `data_split.ipynb` reads them and writes every other file, with seed 42.

```
data/
  raw_data/
    sx_clean_df.csv                 cleaned records, individual questionnaire items (dataset A)
    sx_clean_Total_df.csv           cleaned records, summed questionnaire scores (dataset B)
  build_data/
    x_indv_build.csv   y_indv_build.csv     training plus calibration patients, dataset A
    x_sum_build.csv    y_sum_build.csv      training plus calibration patients, dataset B
  train/
    x_indv_train.csv   y_indv_train.csv
    x_sum_train.csv    y_sum_train.csv
  cal/
    x_indv_calibration.csv   y_indv_calibration.csv
    x_sum_calibration.csv    y_sum_calibration.csv
  test/
    x_indv_test.csv    y_indv_test.csv
    x_sum_test.csv     y_sum_test.csv
```

`data_split.ipynb` also writes the patient indices of each split to `split_indices_indv_seed42.pkl` and `split_indices_sum_seed42.pkl` at the repository root. `pipeline/config.py` names every file in `REPRESENTATIONS`. Both representations hold the same patients in each split.
