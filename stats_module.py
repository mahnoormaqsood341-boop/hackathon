import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.weightstats import ztest

# SECTION 1: BASIC STATISTICS

def calculate_basics(data):
    # Calculate mean, median and mode
    mean = np.mean(data)
    median = np.median(data)
    mode_val = stats.mode(data, keepdims=True).mode
    return mean, median, mode_val

def get_data_dispersion(data):
    # Calculate variance and standard deviation
    data_variance = np.var(data)
    standard_deviation = np.std(data)
    return data_variance, standard_deviation

def get_z_scores(data):
    # Calculate z-score for each value
    return stats.zscore(data)

def get_probability_density(data):
    # Calculate PDF for normal distribution
    mu, sigma = np.mean(data), np.std(data)
    pdf_values = stats.norm.pdf(data, mu, sigma)
    return pdf_values

def get_cdf_values(data):
    # Calculate CDF for normal distribution
    mu, sigma = np.mean(data), np.std(data)
    cdf_values = norm.cdf(data, mu, sigma)
    return cdf_values

# SECTION 2: CLEANING AND OUTLIERS

def check_missing(df):
    # Count missing values and their percentage
    report = df.isnull().sum()
    percent = (report / len(df)) * 100
    return report, percent

def fill_missing(df, column, method="median"):
    # Fill missing values using mean, median, mode or drop rows
    if method == "drop":
        df = df.dropna(subset=[column])
    elif method == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif method == "mode":
        mode_val = df[column].mode()
        if not mode_val.empty:
            df[column] = df[column].fillna(mode_val)
    else:
        df[column] = df[column].fillna(df[column].median())
    return df

def clean_data(df):
    # Remove duplicates and fill all missing values automatically
    if df is None:
        return None, 0, None
    
    # Count and remove duplicates
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates()
    
    # Get missing report
    missing_report = df.isnull().sum()
    
    # Fill missing values based on data type
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                m = df[col].mode()
                if not m.empty:
                    df[col] = df[col].fillna(m)
    
    return df, duplicates, missing_report

def fix_outliers_iqr(df, column):
    # Use IQR method to cap outliers
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

def find_outliers_zscore(data, threshold=3):
    # Find outliers using z-score threshold
    z_scores = np.abs(stats.zscore(data))
    return np.where(z_scores > threshold)

# SECTION 3: HYPOTHESIS AND ANALYSIS

def run_hypothesis_tests(group1, group2):
    # Run t-test and z-test on two groups
    t_stat, t_p = stats.ttest_ind(group1, group2)
    z_stat, z_p = ztest(group1, group2)
    return {"t_test_p": t_p, "z_test_p": z_p}

def run_chi_square_test(df, column1, column2):
    # Run chi-square test for categorical columns
    contingency_table = pd.crosstab(df[column1], df[column2])
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    return chi2, p_val

def get_feature_ranking(df, target_col):
    # Rank top 10 features by correlation with target
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
    return correlations[1:11]

def normalize_target(df, target_column):
    # Normalize target column between 0 and 1
    min_val = df[target_column].min()
    max_val = df[target_column].max()
    df[target_column] = (df[target_column] - min_val) / (max_val - min_val)
    return df

def analyze_categorical_target(df, cat_column, target_column):
    # Find mean of target for each category
    return df.groupby(cat_column)[target_column].mean()

# SECTION 4: MASTER AUTOMATION

def full_eda_automation(df, target):
    # Run all EDA steps at once
    print("Starting Automated EDA...")
    
    # Run cleaning
    df, dupes, missing = clean_data(df)
    
    # Fix outliers in numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df = fix_outliers_iqr(df, col)
        
    # Get target skewness
    target_skew = df[target].skew()
    
    # Get top 10 features
    top_10 = get_feature_ranking(df, target)
    
    print("EDA Completed")
    return df, top_10, target_skew













