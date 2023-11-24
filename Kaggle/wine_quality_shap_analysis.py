import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
shap.initjs()


######################################################################
# Load Data
######################################################################
df_wine = pd.read_csv("WineQT.csv")


######################################################################
# Basic EDA
######################################################################
df_wine.shape

df_wine.columns

df_wine.head()

df_wine.info()

df_wine.describe().T

df_wine.isnull().sum()

df_wine["quality"].unique()


######################################################################
# Multiclass classification
######################################################################

df_wine["quality"] = [1 if x > 5 else 0 for x in df_wine["quality"].values]


plt.figure(figsize=(6, 4))
sns.heatmap(df_wine.corr() > 0.7, annot=True, cmap="YlGnBu")
plt.show()

features = df_wine.drop(["quality", "Id"], axis=1)
target = df_wine["quality"]


######################################################################
# Split into train and test
######################################################################
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=1
)


######################################################################
# Train a machine learning model
######################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = model.predict(X_test)

# Classification Report
print(classification_report(y_pred, y_test))


######################################################################
# Feature importance
######################################################################
importances = model.feature_importances_
feaure_importances = pd.DataFrame(
    importances, index=features.columns, columns=["Feature"]
)
feaure_importances.sort_values(ascending=False, inplace=True, by="Feature")
plt.barh(feaure_importances.index, feaure_importances.Feature)


######################################################################
# Train a machine learning model
######################################################################

explainer = shap.Explainer(model)
shap_values_cat = explainer.shap_values(X_train)

######################################################################
# Visualization
######################################################################
shap.summary_plot(shap_values_cat, X_train)
shap.summary_plot(shap_values_cat[0, :, 0], X_train)
shap.summary_plot(shap_values_cat[0, :, 1], X_train)
shap.plots.force(
    explainer.expected_value[0],
    shap_values_cat[0][0, :],
    X_train.iloc[0, :],
    matplotlib=True,
)

# --------------------------------------------------------------------

explainer = shap.Explainer(model)
shap_values_cat = explainer(X_train)
shap_values_cat.shape

shap.plots.waterfall(shap_values_cat[0, :, 0])
shap.plots.waterfall(shap_values_cat[0, :, 1])

X_train.head(1).T
model.predict_proba(X_train)[1]

shap.plots.bar(shap_values_cat[0, :, 0])
shap.plots.bar(shap_values_cat[0, :, 1])
shap.plots.beeswarm(shap_values_cat)


######################################################################
# Calculate log loss
######################################################################
from sklearn.metrics import accuracy_score, log_loss

y_pred_prob = model.predict_proba(X_train)[:, 1]  # Probability of class 1
log_loss_value = log_loss(y_train, y_pred_prob)
print(f"Log Loss: {log_loss_value:.2f}")

# Calculate average predicted log odds across the dataset
average_log_odds = sum(y_pred_prob) / len(y_pred_prob)
print(f"Average Predicted Log Odds: {average_log_odds:.2f}")


######################################################################
# Multi class classification
######################################################################
df_wine = pd.read_csv("WineQT.csv")

df_wine.isnull().sum()

df_wine["quality"].unique()

df_wine["quality"] = df_wine["quality"] - 3

for col in df_wine.columns:
    if df_wine[col].isnull().sum() > 0:
        df_wine[col] = df_wine[col].fillna(df_wine[col].mean())


features = df_wine.drop(["quality", "Id"], axis=1)
target = df_wine["quality"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=1
)

# Train a machine learning model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = model.predict(X_train)

# Classification Report
print(classification_report(y_pred, y_train))

explainer = shap.Explainer(model)
shap_values_cat = explainer(X_train)
shap_values_cat.shape

X_train.head(1)

shap.plots.waterfall(shap_values_cat[0, :, 0])
shap.plots.waterfall(shap_values_cat[0, :, 1])
shap.plots.waterfall(shap_values_cat[0, :, 2])
shap.plots.waterfall(shap_values_cat[0, :, 3])
shap.plots.waterfall(shap_values_cat[0, :, 4])
shap.plots.waterfall(shap_values_cat[0, :, 5])


preds = model.predict(X_train)
new_shap_values = []
for i, pred in enumerate(preds):
    new_shap_values.append(shap_values_cat.values[i][:, pred])

shap_values_cat.values = np.array(new_shap_values)
print(shap_values_cat.shape)

shap.summary_plot(shap_values_cat, X_train)


shap.plots.bar(shap_values_cat)

shap.plots.beeswarm(shap_values_cat)
