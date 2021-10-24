import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import LabelEncoder

files = os.listdir("row_dataset")

for CSV_file in files:
    print(CSV_file, end=" - ")
    df = pd.read_csv("row_dataset/" + CSV_file)
    all_names = df.columns
    num_names = list(df.select_dtypes(exclude=['object']).columns)
    non_numeric = [item for item in all_names if item not in num_names]
    for col_name in non_numeric:
        lb_encoder = LabelEncoder()
        df[col_name] = lb_encoder.fit_transform(df[col_name])
    df.to_csv("numeric_row_dataset/" + CSV_file)
