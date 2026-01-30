import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

DATA_PATH = "data/creditcard.csv"
TARGET = "Class"
THRESHOLD = 0.96
MODEL_NAME = "CreditCardFraud_XGB"
PRIMARY_METRIC = "f1_class_1"
RANDOM_STATE = 42

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# -----------------------------
# Load candidate model (latest run)
# -----------------------------
candidate_model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}/None"
)

y_proba = candidate_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= THRESHOLD).astype(int)

candidate_f1 = f1_score(y_test, y_pred)

print(f"Candidate F1 (fraud): {candidate_f1:.4f}")

# -----------------------------
# Fetch previous Production model score
# -----------------------------
client = MlflowClient()

try:
    prod_versions = client.get_latest_versions(
        MODEL_NAME, stages=["Production"]
    )

    prod_run_id = prod_versions[0].run_id
    prod_run = client.get_run(prod_run_id)
    prod_f1 = prod_run.data.metrics.get(PRIMARY_METRIC, 0)

    print(f"Production F1 (fraud): {prod_f1:.4f}")

except Exception:
    # First deployment case
    print("No Production model found. Allowing deployment.")
    prod_f1 = -1

# -----------------------------
# Log evaluation
# -----------------------------
with mlflow.start_run(run_name="evaluation"):
    mlflow.log_metric("candidate_f1_class_1", candidate_f1)
    mlflow.log_metric("production_f1_class_1", prod_f1)
    mlflow.log_param("threshold", THRESHOLD)

# -----------------------------
# Quality Gate
# -----------------------------
if candidate_f1 < prod_f1:
    raise ValueError(
        f"❌ Model rejected: candidate F1 ({candidate_f1:.4f}) "
        f"< production F1 ({prod_f1:.4f})"
    )

print("✅ Model passed quality gate (>= Production)")
