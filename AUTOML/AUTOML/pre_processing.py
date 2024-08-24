import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    RobustScaler,
    LabelEncoder,
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
import utils as utils


class PreProcessing:

    def __init__(self, df_train, df_test, NUM_FEATS, CAT_FEATS, ORDINAL_FEATS, target):
        self.df_train = df_train
        self.df_test = df_test
        self.NUM_FEATS = NUM_FEATS
        self.CAT_FEATS = CAT_FEATS
        self.ORDINAL_FEATS = ORDINAL_FEATS
        self.target = target

    @utils.error_handler
    @utils.measure_time
    def transform_features(self):

        numeric_imputer = SimpleImputer(strategy="median")
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

        numeric_pipeline = Pipeline([("imputer", numeric_imputer)])

        categorical_pipeline = Pipeline(
            [("imputer", categorical_imputer), ("encoder", ordinal_encoder)]
        )

        preprocessor = ColumnTransformer(
            [
                ("num", numeric_pipeline, self.NUM_FEATS),
                ("cat", categorical_pipeline, self.CAT_FEATS),
            ]
        )

        pipeline = Pipeline([("preprocessor", preprocessor)])

        print(f"*" * 50)
        print(f"----------------- Preprocessing Pipeline -----------------")
        print(pipeline)
        print(f"*" * 50)

        df_train_processed = pipeline.fit_transform(self.df_train)
        df_test_processed = pipeline.transform(self.df_test)

        df_train_processed = pd.DataFrame(
            df_train_processed, columns=self.NUM_FEATS + self.CAT_FEATS
        )
        df_train_processed[self.target] = self.df_train[self.target].values
        df_train_processed["id"] = self.df_train["id"].values

        df_test_processed = pd.DataFrame(
            df_test_processed, columns=self.NUM_FEATS + self.CAT_FEATS
        )
        df_test_processed["id"] = self.df_test["id"].values

        le = LabelEncoder()
        df_train_processed[self.target] = le.fit_transform(
            df_train_processed[self.target]
        )

        X = df_train_processed.drop(["id", self.target], axis=1)
        y = df_train_processed[self.target]
        df_test_processed = df_test_processed.drop(["id"], axis=1)

        return df_train_processed, df_test_processed, X, y
