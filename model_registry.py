import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from mlflow.tracking import MlflowClient

# ১. সার্ভার কানেক্ট করা
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()
model_name = "My-Final-Project-Model"

print("--- Step 1: Training and Logging Model ---")

rfc = RandomForestClassifier(n_estimators=10)
with mlflow.start_run(run_name="Thesis_Training_Run") as run:
    run_id = run.info.run_id
   
    mlflow.sklearn.log_model(
        sk_model=rfc, 
        artifact_path="model_files", 
        registered_model_name=model_name
    )

print(f"Model logged with Run ID: {run_id}")

print("\n--- Step 2: Adding Metadata (Tags & Description) ---")

client.update_registered_model(
    name=model_name, 
    description="This model predicts results for my thesis project."
)

client.set_registered_model_tag(model_name, "framework", "sklearn")
client.set_registered_model_tag(model_name, "status", "testing")

print("\n--- Step 3: Setting Alias (Champion) ---")

client.set_registered_model_alias(model_name, "Champion", version="1")
print("Version 1 is now the Champion!")

print("\n--- Step 4: Retrieving Model via Alias ---")

champion_version = client.get_model_version_by_alias(model_name, "Champion")

print(f"Current Champion Version: {champion_version.version}")
print(f"Champion Model Location: {champion_version.source}")
print(f"Tags attached: {champion_version.tags}")

print("\n--- Success: Check MLflow UI at http://localhost:5000 ---")