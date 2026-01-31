import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Library imports
from stats_module import (
    full_eda_automation, 
    analyze_categorical_target, 
    run_chi_square_test, 
    normalize_target,
    calculate_basics
)

# 1. Setting Visualization Style
plt.style.use('default')
sns.set_theme()

def run_analysis():
    # Load Dataset
    df = pd.read_csv('AmesHousing.csv')
    target_col = 'SalePrice'
    
    print(f"--- 1. DATA INSPECTION ---")
    print(f"Dataset Shape: {df.shape}")
    print(df.head()) # Showing first 5 records as requested

    # --- 2. MISSING DATA VISUALIZATION ---
    print("\n--- 2. MISSING DATA MAP ---")
    plt.figure(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Data Map (Yellow lines represent gaps)")
    plt.show()

    # --- 3. TARGET DISTRIBUTION & LOG TRANSFORM ---
    print("\n--- 3. TARGET VARIABLE ANALYSIS ---")
    plt.figure(figsize=(12, 5))
    
    # Original Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df[target_col], kde=True, color='blue')
    plt.title(f"Original {target_col} (Skew: {df[target_col].skew():.2f})")
    
    # Applying Log Transformation (Requirement: Transform target if needed)
    # Log transform skewness ko kam karke distribution ko normal banata hai
    df[target_col] = np.log1p(df[target_col])
    
    # Transformed Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(df[target_col], kde=True, color='green')
    plt.title(f"Log Transformed {target_col} (Skew: {df[target_col].skew():.2f})")
    plt.show()

    # --- 4. AUTOMATED CLEANING & OUTLIERS ---
    # Running your automated library function
    clean_df, top_features, target_skew = full_eda_automation(df, target_col)

    # --- 5. CATEGORICAL & STATISTICAL ANALYSIS ---
    style_impact = analyze_categorical_target(clean_df, 'House Style', target_col)
    chi_stat, p_val = run_chi_square_test(clean_df, 'Central Air', 'Paved Drive')

    # --- 6. VISUALIZATION SECTION ---
    print("\n Generating Relationship Visualizations...")

    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    top_cols = top_features.index.tolist() + [target_col]
    sns.heatmap(clean_df[top_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Top 10 Features Correlation Matrix")
    plt.show()

    # Scatter Plot with Regression Line
    plt.figure(figsize=(8, 5))
    sns.regplot(data=clean_df, x='Gr Liv Area', y=target_col, scatter_kws={'alpha':0.5}, line_kws={"color": "red"})
    plt.title("Living Area vs Sale Price (Relationship with Regression)")
    plt.show()

    # --- 7. FINAL REPORT & PROBABILITY ---
    print("\n" + "="*40)
    print("      FINAL SMART EDA INSIGHTS      ")
    print("="*40)
    
    # Basic probability calculation (Requirement)
    # Price threshold 12.5 (log value) ke upar ghar milne ka chance
    prob_high_value = len(clean_df[clean_df[target_col] > 12.5]) / len(clean_df)
    
    print(f"Target Skewness (Post-Transform): {target_skew:.4f}")
    print(f"Chi-Square P-Value (AC vs Paved Drive): {p_val:.4f}")
    print(f"Probability of High-Value Houses: {prob_high_value:.2%}")
    
    print("\nTop 10 Most Important Features:")
    print(top_features)
    
    # Statistical Summary for Target (Requirement)
    mean, median, mode = calculate_basics(clean_df[target_col])
    print(f"\nFinal Target Stats -> Mean: {mean:.2f}, Median: {median:.2f}")
    
    print("\nAnalysis Finished Successfully!")

if __name__ == "__main__":
    run_analysis()


