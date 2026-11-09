"""Preprocessing V2 for the UNSW-NB15 scalability study (implementation only; no model code).

Locked specification (approved by the project owner):
  * predictors: 39 numerical + 3 categorical (proto, service, state)
  * excluded from the input: id (sample identity / traceability only), attack_cat, label (target only)
  * categorical vocabulary: fixed once from the COMPLETE TRAINING PARTITION ONLY (never from test rows)
  * one-hot with handle_unknown = "ignore": an unseen value gives an all-zero block for that variable;
    the row is kept and the dimension does not change
  * numerical scaling: StandardScaler fitted ONLY on the actual training sample of a run;
    the same scaler is used for every method in that cell and for the corresponding test transform
  * one-hot indicators are NOT scaled
  * zero-variance numerical features are NOT dropped; the documented StandardScaler behaviour
    (scale_ = 1.0 for zero variance) is used and the feature names are recorded
  * canonical feature order: 39 numerical (raw schema order), then encoded proto, service, state;
    inside each categorical variable the categories are in sorted (Unicode code point) order
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXCLUDED = ["id", "attack_cat", "label"]
CATEGORICAL = ["proto", "service", "state"]


def read_partition(path):
    """Read a CSV without silently converting tokens such as 'NA' or 'None' to missing values."""
    return pd.read_csv(path, keep_default_na=False)


def numerical_columns(df):
    """Numerical predictors in raw schema order."""
    return [c for c in df.columns if c not in EXCLUDED and c not in CATEGORICAL]


def build_vocabulary(vocab_source_df):
    """Category values per variable, sorted. `vocab_source_df` must be the training data only."""
    return {c: sorted(vocab_source_df[c].astype(str).unique().tolist()) for c in CATEGORICAL}


class PreprocessorV2:
    def __init__(self, numerical, vocabulary):
        self.numerical = list(numerical)
        self.vocabulary = {c: list(vocabulary[c]) for c in CATEGORICAL}
        self.scaler = None
        self.encoder = OneHotEncoder(
            categories=[self.vocabulary[c] for c in CATEGORICAL],
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
        # With explicit categories the encoder is fitted on vocabulary values only (never on data rows).
        self.encoder.fit(np.array([[self.vocabulary[c][0] for c in CATEGORICAL]], dtype=object))

    @property
    def feature_names(self):
        names = list(self.numerical)
        for c in CATEGORICAL:
            names += [f"{c}={v}" for v in self.vocabulary[c]]
        return names

    @property
    def dimension(self):
        return len(self.numerical) + sum(len(self.vocabulary[c]) for c in CATEGORICAL)

    def fit_scaler(self, training_sample_df):
        """Fit the numerical scaler on the training sample of the run ONLY."""
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.scaler.fit(training_sample_df[self.numerical].astype("float64").to_numpy())
        return self

    def fit_parameters(self):
        return {
            "scaler_mean": self.scaler.mean_.copy(),
            "scaler_scale": self.scaler.scale_.copy(),
            "scaler_var": self.scaler.var_.copy(),
        }

    def zero_variance_features(self):
        return [n for n, v in zip(self.numerical, self.scaler.var_) if v == 0.0]

    def transform(self, df):
        num = self.scaler.transform(df[self.numerical].astype("float64").to_numpy())
        cat = self.encoder.transform(df[CATEGORICAL].astype(str).to_numpy(dtype=object))
        return np.ascontiguousarray(np.hstack([num, cat]), dtype=np.float64)


def manual_onehot(df, vocabulary):
    """Independent implementation of the same encoding (used only to cross-check the encoder)."""
    blocks = []
    for c in CATEGORICAL:
        idx = {v: i for i, v in enumerate(vocabulary[c])}
        vals = df[c].astype(str).to_numpy()
        m = np.zeros((len(df), len(vocabulary[c])), dtype=np.float64)
        for r, v in enumerate(vals):
            j = idx.get(v)
            if j is not None:
                m[r, j] = 1.0
        blocks.append(m)
    return np.hstack(blocks)
