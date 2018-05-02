# For testing the dependency 
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os
from pathlib import Path
print("import ready")

cwd = os.getcwd()

f = cwd + "\\input\\test.csv"
# f = cwd + "/input/test.csv" # for UNIX

print(f)

bl = Path(f).is_file()

print(bl)