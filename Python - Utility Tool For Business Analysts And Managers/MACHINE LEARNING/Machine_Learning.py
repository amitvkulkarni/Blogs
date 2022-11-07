# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------- Importing libraries---------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix, roc_curve
from sklearn.metrics import plot_roc_curve
# import seaborn as sns

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------- Data Loading---------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("SAheart.csv")

# Features & Labels
X_features = df.drop("chd", axis=1)
y_labels = df["chd"]

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------- Setting up Train and Test datasets---------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_labels, test_size=0.3, random_state=42
)

numeric_features = (
    df.select_dtypes(include=["int64", "float64"]).columns
)

categorical_features = (
    df.select_dtypes(include=["object"]).drop(["chd", 'famhist'], axis=1).columns
)


numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=2)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


rf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Building multiple models with same pipeline
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
classifiers = [KNeighborsClassifier(3), RandomForestClassifier(), LogisticRegression()]
model_res = []
for classifier in classifiers:
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    pipe.fit(X_train, y_train)  
    plot_roc_curve(pipe, X_train, y_train)
    plt.savefig('./img/ROC_AUC.png')
                 
    model_res.append(
        {
            "Model_Name": classifier,
            "Accuracy": round(pipe.score(X_test, y_test),5),
            "F1_Score": round(f1_score(y_test, y_pred, average="macro"),5),
            "Precision": round(precision_score(y_test, y_pred, average="macro"),5),
            "Recall": round(recall_score(y_test, y_pred, average="macro"),5),
        
        }
  
    )
    
df_res = pd.DataFrame(model_res, columns=['Model_Name', 'Accuracy', 'F1_Score','Precision', 'Recall'])
Executive_Summary = f"All the models are built using python. In this report, the analysis is carried out using {classifiers[0]}, {classifiers[1]} and {classifiers[2]}. The model metrics are as below"


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------- Documentation using docxtpl---------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import datetime as dt
from docxtpl import DocxTemplate, InlineImage

# create a document object
doc = DocxTemplate("TEMPLATE_MACHINE_LEARNING_REPORT.docx")


context = {
    "Executive_Summary": Executive_Summary,
    "model_res": model_res, 
    "ROC_AUC_Curve": InlineImage(doc, "img/ROC_AUC.png"),

}

# render context into the document object
doc.render(context)
todayStr = dt.datetime.now().strftime("%d-%b-%Y")

# save the document object as a word file
reportWordPath = "./OUTPUT/MACHINE LEARNING MODEL RESULTS_{0}.docx".format(todayStr)
doc.save(reportWordPath)

print('Machine Learning Model Report Successfully Generated !!!!')
