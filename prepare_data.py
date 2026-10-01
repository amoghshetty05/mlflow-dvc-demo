import os
import pandas as pd
from sklearn import datasets

os.makedirs("data", exist_ok=True)
datasets.load_iris(as_frame=True).frame.to_csv("data/train.csv", index=False)
