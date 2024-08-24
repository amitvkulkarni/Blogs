import pandas as pd
import numpy as np
import time
from AUTOML.chain_stages import ChainStages


####################################################################################
# Data Loading
####################################################################################

# df_train = pd.read_csv("data/Poisonous_Mushrooms_train.csv")
# df_train = df_train.head(10000)

# df_test = pd.read_csv("data/Poisonous_Mushrooms_test.csv")
# df_test = df_test.head(10000)

df_train = pd.read_csv("data/Machine_Failure_train.csv")
df_train = df_train.head(10000)

df_test = pd.read_csv("data/Machine_Failure_test.csv")
df_test = df_test.head(10000)


####################################################################################
# Configurations
####################################################################################

target = "Machine failure"
# target = "class"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_TRAILS = 10

ORDINAL_FEATS = []
NUM_FEATS = list(df_train.select_dtypes(include="float").columns)
print(f"Numerical features -> {NUM_FEATS}")
print("-" * 150)

CAT_FEATS = [
    x for x in df_train.columns if x not in NUM_FEATS and x != target and x != "id"
]
print(f"Categorical features -> {CAT_FEATS}")
print("-" * 150)

DROP_COLUMNS = ["id"]


####################################################################################
# Initiate model building
####################################################################################

cs = ChainStages(
    df_train, df_test, NUM_FEATS, CAT_FEATS, ORDINAL_FEATS, target, 1, 3, DROP_COLUMNS
)
model_summary = cs.execute_stages()
model_summary
