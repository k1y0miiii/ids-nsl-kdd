"""src/data.py — download, load, and preprocess NSL-KDD."""

import os
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TRAIN_URL = (
    "https://raw.githubusercontent.com/tuanio/nsl_kdd/main/KDDTrain+.txt"
)
TEST_URL = (
    "https://raw.githubusercontent.com/tuanio/nsl_kdd/main/KDDTest+.txt"
)

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

CATEGORICAL = ["protocol_type", "service", "flag"]

DOS_LABELS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "apache2", "udpstorm", "processtable", "worm",
}
PROBE_LABELS = {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"}
R2L_LABELS = {
    "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
    "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
    "snmpguess", "xlock", "xsnoop", "httptunnel",
}
U2R_LABELS = {
    "buffer_overflow", "loadmodule", "perl", "rootkit",
    "ps", "sqlattack", "xterm",
}


def _label_to_category(label: str) -> str:
    if label == "normal":
        return "normal"
    if label in DOS_LABELS:
        return "dos"
    if label in PROBE_LABELS:
        return "probe"
    if label in R2L_LABELS:
        return "r2l"
    if label in U2R_LABELS:
        return "u2r"
    # Unknown attack — treat as generic attack for safety
    return "unknown"


def download_data() -> None:
    """Download KDDTrain+ and KDDTest+ if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for url, filename in [(TRAIN_URL, "KDDTrain+.txt"), (TEST_URL, "KDDTest+.txt")]:
        dest = os.path.join(DATA_DIR, filename)
        if os.path.exists(dest):
            print(f"[data] {filename} already exists, skipping download.")
            continue
        print(f"[data] Downloading {filename} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"[data] Saved {filename} ({len(r.content) // 1024} KB)")


def _load_raw(filename: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["label"].apply(_label_to_category)
    df["binary"] = (df["label"] != "normal").astype(int)  # 0=normal, 1=attack
    return df


def load_data(
    download: bool = True,
):
    """
    Returns preprocessed train/test splits as a 7-tuple:
      X_train, X_test, y_train_binary, y_test_binary,
      y_train_multi, y_test_multi, feature_names

    Preprocessing steps:
      1. Drop difficulty_level column.
      2. One-hot encode protocol_type, service, flag (drop_first=False so all
         categories are explicit).
      3. StandardScaler on numeric columns.
      4. Return both binary (0/1) and multiclass (normal/dos/probe/r2l/u2r)
         targets separately.
    """
    if download:
        download_data()

    train_raw = _load_raw("KDDTrain+.txt")
    test_raw = _load_raw("KDDTest+.txt")

    train_raw = _build_targets(train_raw)
    test_raw = _build_targets(test_raw)

    y_train_binary = train_raw["binary"]
    y_test_binary = test_raw["binary"]
    y_train_multi = train_raw["category"]
    y_test_multi = test_raw["category"]

    # Drop label columns
    drop_cols = ["label", "difficulty_level", "binary", "category"]
    X_train = train_raw.drop(columns=drop_cols)
    X_test = test_raw.drop(columns=drop_cols)

    # One-hot encode categoricals
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL, drop_first=False)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL, drop_first=False)

    # Align columns (test may have fewer category values)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Scale numerics (non-dummy columns)
    numeric_cols = [
        c for c in X_train.columns
        if not any(c.startswith(cat + "_") for cat in CATEGORICAL)
    ]
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    feature_names = list(X_train.columns)

    print(f"[data] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[data] Binary train dist:\n{y_train_binary.value_counts()}")
    print(f"[data] Multi train dist:\n{y_train_multi.value_counts()}")

    return X_train, X_test, y_train_binary, y_test_binary, y_train_multi, y_test_multi, feature_names


if __name__ == "__main__":
    load_data()
