import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae

# prepare dataset
data = load_specific_dataset("physionet_2012")
X = data["X"]
num_samples = len(X["RecordID"].unique())
X = X.drop(["RecordID", "Time"], axis=1)
X = StandardScaler().fit_transform(X.to_numpy())
X = X.reshape(num_samples, 48, -1)
X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1)
# set the missing values to np.nan
X = masked_fill(X, 1 - missing_mask, np.nan)
dataset = {"X": X}
print(dataset["X"].shape)  # (11988, 48, 37), 11988 samples, 48 time steps, 37 features

# initialize the model
saits = SAITS(
    n_steps=48,
    n_features=37,
    n_layers=2,
    d_model=256,
    d_inner=128,
    n_heads=4,
    d_k=64,
    d_v=64,
    dropout=0.1,
    epochs=10,
    saving_path="EHR_results/saits",
)

# train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
saits.fit(dataset)
# impute the originally-missing values and artificially-missing values
imputation = saits.impute(dataset)
# calculate mean absolute error on the ground truth (artificially-missing values)
mae = cal_mae(imputation, X_intact, indicating_mask)