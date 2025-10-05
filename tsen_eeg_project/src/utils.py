import pandas as pd

def read_labels(labels_csv: str):
    df = pd.read_csv(labels_csv)
    # Expect columns: filename, label
    return df
