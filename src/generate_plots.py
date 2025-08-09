import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Data from previous step
data = {
    "Sentiment-Enhanced": {
        "LightGBM": {"test_accuracy": 0.75, "up_accuracy": 1.0, "down_accuracy": 0.25},
        "RandomForest": {"test_accuracy": 0.527, "up_accuracy": 0.583, "down_accuracy": 0.416},
        "XGBoost": {"test_accuracy": 0.5, "up_accuracy": 0.416, "down_accuracy": 0.666},
    },
    "Baseline": {
        "LightGBM": {"test_accuracy": 0.606, "up_accuracy": 0.652, "down_accuracy": 0.5},
        "RandomForest": {"test_accuracy": 0.393, "up_accuracy": 0.304, "down_accuracy": 0.6},
        "XGBoost": {"test_accuracy": 0.333, "up_accuracy": 0.173, "down_accuracy": 0.7},
    },
    "Simple": {
        "LogisticRegression": {"test_accuracy": 0.333, "up_accuracy": 0.0, "down_accuracy": 1.0},
    }
}

# Prepare data for plotting
plot_data = []
for category, models in data.items():
    for model_name, metrics in models.items():
        plot_data.append({
            "Model": f"{category} {model_name}",
            "Category": category,
            "Test Accuracy": metrics["test_accuracy"],
            "Up Days Accuracy": metrics["up_accuracy"],
            "Down Days Accuracy": metrics["down_accuracy"],
        })

df_plot = pd.DataFrame(plot_data)

# Sort by Test Accuracy for better visualization
df_plot = df_plot.sort_values(by="Test Accuracy", ascending=False)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Plot 1: Overall Test Accuracy Comparison
plt.figure(figsize=(12, 7))
sns.barplot(x="Model", y="Test Accuracy", data=df_plot, palette="viridis")
plt.title("Overall Test Accuracy Comparison of Models", fontsize=16)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("models/overall_test_accuracy.png")
plt.close()

# Plot 2: Directional Accuracy Comparison
df_directional = df_plot.melt(id_vars=["Model", "Category"], value_vars=["Up Days Accuracy", "Down Days Accuracy"], var_name="Directional Accuracy Type", value_name="Accuracy")

plt.figure(figsize=(14, 8))
sns.barplot(x="Model", y="Accuracy", hue="Directional Accuracy Type", data=df_directional, palette="muted")
plt.title("Directional Accuracy (Up vs. Down Days) Comparison", fontsize=16)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1.1) # Slightly above 1 to show 1.0 clearly
plt.xticks(rotation=45, ha="right")
plt.legend(title="Accuracy Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("models/directional_accuracy_comparison.png")
plt.close()

print("Plots generated and saved to 'models/' directory.")
