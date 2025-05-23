import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_all_results():
    """Load results from all models"""
    models = ['random_forest', 'logistic_regression', 'svm', 'gradient_boosting', 'xgboost']
    all_results = []
    
    for model in models:
        model_dir = f'model_results/{model}'
        if os.path.exists(model_dir):
            # Load First_Weight results
            fw_results = pd.read_csv(
                os.path.join(model_dir, f'{model}_first_weight_only_results.csv')
            )
            # Load Other Features results
            other_results = pd.read_csv(
                os.path.join(model_dir, f'{model}_other_features_results.csv')
            )
            all_results.extend([fw_results, other_results])
    
    return pd.concat(all_results, ignore_index=True)

def create_comparison_plots(results_df, output_dir):
    """Visualize model comparisons"""
    # Accuracy comparison plots
    plt.figure(figsize=(12, 6))
    metrics = ['Best_Score', 'Mean_CV_Score', 'Test_Accuracy']
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        sns.barplot(
            data=results_df,
            x='Model',
            y=metric,
            hue='Feature_Set'
        )
        plt.xticks(rotation=45)
        plt.title(f'{metric} Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_metrics.png'))
    plt.close()

def main():
    # Create directory for saving results
    output_dir = 'comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all model results
    all_results = load_all_results()
    
    # Compare and visualize results
    create_comparison_plots(all_results, output_dir)
    
    # Save detailed comparison results to a CSV file
    all_results.to_csv(
        os.path.join(output_dir, 'all_models_comparison.csv'),
        index=False
    )
    
    # Identify the best model
    best_model = all_results.loc[all_results['Test_Accuracy'].idxmax()]
    
    # Save summary of results to a text file
    with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
        f.write("=== Model Comparison Summary ===\n\n")
        f.write(f"Best performing model: {best_model['Model']}\n")
        f.write(f"Feature set: {best_model['Feature_Set']}\n")
        f.write(f"Test accuracy: {best_model['Test_Accuracy']:.3f}\n")
        f.write(f"Cross-validation score: {best_model['Mean_CV_Score']:.3f}\n")
        
        f.write("\nAll models performance summary:\n")
        f.write(all_results.groupby(['Model', 'Feature_Set'])['Test_Accuracy'].mean().to_string())
    
    print("\n=== Comparison completed. Results saved in 'comparison_results' directory ===")

if __name__ == '__main__':
    main()