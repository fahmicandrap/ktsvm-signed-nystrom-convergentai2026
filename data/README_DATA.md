# Data

The UNSW-NB15 partitions are not redistributed in this repository. Download them from the official UNSW-NB15 page, https://research.unsw.edu.au/projects/unsw-nb15-dataset (Cyber Range Lab, UNSW Canberra). The data set is free for academic research; commercial use must be agreed with its authors, and users must cite Moustafa & Slay (2015, 2016).

Verify the two files used in this study:

| File | Rows | SHA-256 |
|---|---|---|
| UNSW_NB15_training-set.csv | 175,341 | bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa |
| UNSW_NB15_testing-set.csv | 82,332 | 734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559 |

`sample_composition.csv` lists, for each training sample, the class counts, the number of unique predictor vectors (42 raw predictor columns) and the SHA-256 of its sampling manifest.
