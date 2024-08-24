import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    RocCurveDisplay,
    auc,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


from sklearn.inspection import permutation_importance
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import matthews_corrcoef
import utils as utils


# target = "class"
TEST_SIZE = 0.2
RANDOM_STATE = 42


class BuildModels:

    def __init__(self, X, y, models, show_plots=True):
        self.X = X
        self.y = y
        self.models = models
        self.show_plots = show_plots

    @utils.error_handler
    @utils.measure_time
    def build_baseline_models(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        metrics = []

        # Train and evaluate models
        for name, model in self.models.items():
            # print("**" * 30)
            model_name = f"{type(model).__name__}"
            # print(f"Initiating model building -- {model_name}")
            # print("**" * 30)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            # Compute ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            metrics.append(
                {
                    "Model": type(model).__name__,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "AUC": roc_auc,
                    "MCC": mcc,
                }
            )
            if self.show_plots:
                print(
                    f"Classification Report:\n{classification_report(y_test, y_pred)}\n"
                )

                # Create subplots for ROC curve and feature importance
                fig, axs = plt.subplots(1, 2, figsize=(18, 6))

                # Plot ROC curve
                axs[0].plot(fpr, tpr, color="blue", lw=2, label="ROC curve")
                axs[0].plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
                axs[0].set_xlabel("False Positive Rate")
                axs[0].set_ylabel("True Positive Rate")
                axs[0].set_title(f"ROC Curve --> {model_name}")
                axs[0].legend(loc="lower right")
                axs[0].grid(True)
                axs[0].text(0.6, 0.2, "AUC = %0.5f" % roc_auc, fontsize=12, ha="center")

                # Plot feature importance
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                else:
                    result = permutation_importance(
                        model, X_test, y_test, n_repeats=10, random_state=42
                    )
                    importances = result.importances_mean

                indices = np.argsort(importances)
                features = X_train.columns[indices]

                gradient = np.linspace(0, 1, len(importances))
                color_map = plt.get_cmap("viridis")
                colors = color_map(gradient)

                axs[1].barh(
                    range(X_train.shape[1]),
                    importances[indices],
                    color=colors,
                    align="center",
                )
                axs[1].set_yticks(range(X_train.shape[1]))
                axs[1].set_yticklabels(features)
                axs[1].set_xlabel("Importance")
                axs[1].set_title(f"Feature Importances --> {model_name}")

                plt.tight_layout()
                plt.show()

        metrics_df = pd.DataFrame(metrics)
        return metrics_df
