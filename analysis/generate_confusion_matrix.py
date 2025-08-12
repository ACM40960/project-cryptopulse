import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

def generate_confusion_matrix():
    """
    Loads prediction data from a JSON file, generates a confusion matrix,
    and saves it as a high-quality plot.
    """
    # Load the results data
    with open('models/final_comprehensive_results.json', 'r') as f:
        data = json.load(f)

    # Extract the relevant lists for the LightGBM model
    y_test = data['LightGBM']['predictions']['y_test']
    y_pred = data['LightGBM']['predictions']['y_pred']

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create a dataframe for better labeling
    cm_df = pd.DataFrame(cm,
                         index=['Actual Down', 'Actual Up'],
                         columns=['Predicted Down', 'Predicted Up'])

    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 16}) # Font size for annotations
    plt.title('Confusion Matrix for LightGBM Model', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    # Save the figure
    output_path = 'analysis/visualizations/plots/confusion_matrix.png'
    plt.savefig(output_path, dpi=300) # High resolution
    print(f"Confusion matrix plot saved to {output_path}")

if __name__ == '__main__':
    generate_confusion_matrix()
