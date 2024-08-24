import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import AUTOML.utils as utils

import utils as utils


class DataAnalysis:

    # @staticmethod
    # def measure_time(func):
    #     def wrapper(*args, **kwargs):
    #         start_time = time.time()
    #         result = func(*args, **kwargs)
    #         end_time = time.time()
    #         print(
    #             f"{'-'*20} Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds {'-'*20}"
    #         )
    #         return result

    #     return wrapper

    # @staticmethod
    # def error_handler(func):
    #     def wrapper(*args, **kwargs):
    #         try:
    #             return func(*args, **kwargs)
    #         except Exception as e:
    #             print(f"Error in {func.__name__}: {e}")

    #     return wrapper

    def __init__(
        self,
        df_train,
        df_test,
        NUM_FEATS,
        CAT_FEATS,
        ORDINAL_FEATS,
        target,
        NROWS=3,
        NCOLS=3,
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
    def plot_numeric_features(self):
        fig, axes = plt.subplots(nrows=self.NROWS, ncols=self.NCOLS, figsize=(10, 4))
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        c = ["#90A6B1", "#037d97"]

        for i, var in enumerate(self.NUM_FEATS):
            sns.boxplot(y=self.df_train[var], palette=c, ax=axes[i])
            axes[i].set_title(f"Box plot of {var}")

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    @utils.error_handler
    @utils.measure_time
    def target_distribution(self):
        value_counts = self.df_train[self.target].value_counts()
        percentages = self.df_train[self.target].value_counts(normalize=True) * 100
        percentages = percentages.map(lambda x: f"{x:.2f}%")
        result = pd.DataFrame({"Count": value_counts, "Percentage": percentages})
        return result

    @utils.error_handler
    @utils.measure_time
    def plot_categorical_target_features(self):
        fig, axes = plt.subplots(len(self.CAT_FEATS), 2, figsize=(20, 15))
        for i, column in enumerate(self.CAT_FEATS):
            sns.countplot(x=column, data=self.df_train, ax=axes[i, 0])
            axes[i, 0].set_title(f"{column} Count")

            sns.countplot(x=column, hue=self.target, data=self.df_train, ax=axes[i, 1])
            axes[i, 1].set_title(f"{column} Count with {self.target} Hue")

        sns.despine()
        plt.tight_layout()
        plt.show()

    @utils.error_handler
    @utils.measure_time
    def plot_categorical_dist(self):
        # Define the number of subplots needed
        num_features = len(self.CAT_FEATS)
        num_cols = 4  # Number of columns in subplot grid
        num_rows = (num_features + num_cols - 1) // num_cols

        # Create subplots
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(15, num_rows * 5), constrained_layout=True
        )
        axes = axes.flatten()  # Flatten to make indexing easier

        # Plot each categorical feature
        for i, column in enumerate(self.CAT_FEATS):
            sns.countplot(
                data=self.df_train,
                x=column,
                ax=axes[i],
                order=self.df_train[column].value_counts().index,
            )
            axes[i].set_title(f"Distribution of {column}")
            axes[i].set_xlabel(column)
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis="x", rotation=45)  # Rotate x labels if needed

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.show()

    @utils.error_handler
    @utils.measure_time
    def display_categorical_values_table(self):
        train_categorical_columns = self.df_train.select_dtypes(
            include="object"
        ).columns
        test_categorical_columns = self.df_test.select_dtypes(include="object").columns

        data = []
        for column in train_categorical_columns:
            train_unique_values = self.df_train[column].nunique()
            test_unique_values = (
                self.df_test[column].nunique()
                if column in test_categorical_columns
                else None
            )

            different_levels = False
            if test_unique_values is not None:
                train_levels = set(self.df_train[column].unique())
                test_levels = set(self.df_test[column].unique())
                different_levels = train_levels.symmetric_difference(test_levels)

            data.append(
                [column, train_unique_values, test_unique_values, different_levels]
            )

        print("Unique values for categorical features:")
        df = pd.DataFrame(
            data,
            columns=[
                "Feature",
                "Train_Unique_Values",
                "Test_Unique_Values",
                "Different_Levels",
            ],
        )
        return df

    @utils.error_handler
    @utils.measure_time
    def identify_feature_types(self):
        pd.set_option("display.max_colwidth", None)
        data = {
            "Description": [
                "Total Rows",
                "Total Columns",
                "Numerical Features",
                "Categorical Features",
            ],
            "Count": [
                self.df_train.shape[0],
                self.df_train.shape[1],
                len(self.NUM_FEATS),
                len(self.CAT_FEATS),
            ],
            "Features": [
                (
                    ", ".join(self.NUM_FEATS)
                    if "Numerical" in desc
                    else ", ".join(self.CAT_FEATS) if "Categorical" in desc else None
                )
                for desc in [
                    "Total Rows",
                    "Total Columns",
                    "Numerical Features",
                    "Categorical Features",
                ]
            ],
        }

        df_visual = pd.DataFrame(data)
        return df_visual

    @utils.error_handler
    @utils.measure_time
    def get_correlation(self):
        cor_mat = self.df_train.drop(columns=["id"]).corr(method="pearson")
        mask = np.triu(np.ones_like(cor_mat))

        plt.figure(figsize=(10, 8))
        sns.heatmap(cor_mat, cmap="coolwarm", fmt=".2f", annot=True, mask=mask)
        plt.show()

    @utils.error_handler
    @utils.measure_time
    def get_null_analysis(self):
        train_missing = self.df_train.isnull().sum()
        test_missing = self.df_test.isnull().sum()

        combined_df = pd.DataFrame(
            {"Train Missing Values": train_missing, "Test Missing Values": test_missing}
        )
        return combined_df

    @utils.error_handler
    @utils.measure_time
    def get_outlier_analysis(self):
        outliers = {}
        df = self.df_train[self.NUM_FEATS].copy()
        for col in df.columns:
            v = df[col]
            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_count = ((v < lower_bound) | (v > upper_bound)).sum()
            perc = outliers_count * 100.0 / len(df)
            outliers[col] = (perc, outliers_count)
            if perc != 0:
                print(
                    f"{col} outliers = {perc:.2f}% ({outliers_count} out of {len(df)})"
                )

    @utils.error_handler
    @utils.measure_time
    def auto_eda(self):
        self.plot_numeric_features()
        self.plot_categorical_target_features()
        self.plot_categorical_dist()

        print("Auto EDA Complete")
