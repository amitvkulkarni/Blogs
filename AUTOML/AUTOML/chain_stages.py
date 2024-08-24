import pandas as pd
import numpy as np
import time

# Models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import VotingClassifier, StackingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from AUTOML.data_analysis import DataAnalysis
from AUTOML.pre_processing import PreProcessing
from AUTOML.baseline_models import BuildModels
import utils as utils


class ChainStages:

    def __init__(
        self,
        df_train,
        df_test,
        NUM_FEATS,
        CAT_FEATS,
        ORDINAL_FEATS,
        target,
        NROWS,
        NCOLS,
        DROP_COLUMNS=None,
    ):
        self.df_train = df_train
        self.df_test = df_test
        self.NUM_FEATS = NUM_FEATS
        self.CAT_FEATS = CAT_FEATS
        self.ORDINAL_FEATS = ORDINAL_FEATS
        self.target = target
        self.NROWS = NROWS
        self.NCOLS = NCOLS
        self.DROP_COLUMNS = DROP_COLUMNS

    @utils.error_handler
    @utils.measure_time
    def execute_stages(self):

        eda = DataAnalysis(
            df_train=self.df_train,
            df_test=self.df_test,
            NUM_FEATS=self.NUM_FEATS,
            CAT_FEATS=self.CAT_FEATS,
            ORDINAL_FEATS=self.ORDINAL_FEATS,
            target=self.target,
            NROWS=1,
            NCOLS=3,
            DROP_COLUMNS=self.DROP_COLUMNS,
        )

        # eda.auto_eda()

        process_data = PreProcessing(
            df_train=self.df_train,
            df_test=self.df_train,
            NUM_FEATS=self.NUM_FEATS,
            CAT_FEATS=self.CAT_FEATS,
            ORDINAL_FEATS=self.ORDINAL_FEATS,
            target=self.target,
        )
        df_train_processed, df_test_processed, X, y = process_data.transform_features()

        models = {
            "XGBoost": XGBClassifier(
                objective="binary:logistic", enable_categorical=True
            ),
            "LightGBM": LGBMClassifier(metric="auc", verbose=-1),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=20, random_state=42),
            "DecisionTree": DecisionTreeClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "BalancedRandomForest": BalancedRandomForestClassifier(),
            "MLPClassifier": MLPClassifier(max_iter=1000),
            "HistGradientBoosting": HistGradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "GaussianNB": GaussianNB(),
            "KNeighbors": KNeighborsClassifier(),
            "BernoulliNB": BernoulliNB(),
            "SVC": SVC(probability=True),
        }

        base_models = BuildModels(X, y, models, show_plots=True)
        res = base_models.build_baseline_models()
        return res.style.apply(
            lambda x: (
                utils.highlight_max(x) if x.name in res.columns[1:] else [""] * len(x)
            ),
            axis=0,
        )
