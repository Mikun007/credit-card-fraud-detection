import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load the Data
DATA_PATH = "data/creditcard.csv"
TARGET = "Class"
RANDOM_STATE = 42

# Read The data
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# Pipeline fitting the model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    ))
])

# Setting up MLflow
mlflow.set_experiment("CreditCard_Fraud_XGB")

with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("smote", True)

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="CreditCardFraud_XGB"
    )
