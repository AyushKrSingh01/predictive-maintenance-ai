import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv("data/ai4i2020.csv")

# Same features used during training
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Load trained pipeline
pipeline = joblib.load(
    "models/predictive_maintenance_model.pkl"
)

# Get the Random Forest model
model = pipeline.named_steps["model"]

# Get preprocessing step
preprocessor = pipeline.named_steps["preprocessor"]

# Get feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

# Get feature importance
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df.to_string(index=False))

# Plot
plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png"
)

plt.close()

print("\nFeature importance plot saved successfully.")