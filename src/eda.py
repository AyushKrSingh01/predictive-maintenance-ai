import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/ai4i2020.csv")

# Create output directory
import os
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 1. Failure Distribution
# -----------------------------
plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Machine failure"
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Machines")

plt.tight_layout()
plt.savefig("outputs/failure_distribution.png")
plt.close()


# -----------------------------
# 2. Torque vs Failure
# -----------------------------
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Machine failure",
    y="Torque [Nm]"
)

plt.title("Torque vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Torque [Nm]")

plt.tight_layout()
plt.savefig("outputs/torque_vs_failure.png")
plt.close()


# -----------------------------
# 3. Rotational Speed vs Failure
# -----------------------------
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Machine failure",
    y="Rotational speed [rpm]"
)

plt.title("Rotational Speed vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Rotational Speed [rpm]")

plt.tight_layout()
plt.savefig("outputs/speed_vs_failure.png")
plt.close()


# -----------------------------
# 4. Tool Wear vs Failure
# -----------------------------
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Machine failure",
    y="Tool wear [min]"
)

plt.title("Tool Wear vs Machine Failure")
plt.xlabel("Machine Failure")
plt.ylabel("Tool Wear [min]")

plt.tight_layout()
plt.savefig("outputs/tool_wear_vs_failure.png")
plt.close()


# -----------------------------
# 5. Correlation Matrix
# -----------------------------
numeric_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png")
plt.close()


print("EDA completed successfully.")
print("Plots saved inside outputs/")