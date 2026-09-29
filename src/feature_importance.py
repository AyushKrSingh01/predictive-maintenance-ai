import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_csv("data/ai4i2020.csv")


features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


pipeline = joblib.load(
    "models/predictive_maintenance_model.pkl"
)


model = pipeline.named_steps["model"]


preprocessor = pipeline.named_steps["preprocessor"]


feature_names = preprocessor.get_feature_names_out()


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