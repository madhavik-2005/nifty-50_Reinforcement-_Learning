import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('preprocessed_nifty_sentiment.csv')

print("="*60)
print("INITIAL DATASET ANALYSIS")
print("="*60)
print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nData types:\n{df.dtypes.value_counts()}")

# ============================================
# STEP 1: HANDLE MISSING & INITIAL VALUES
# ============================================
print("\n" + "="*60)
print("STEP 1: HANDLING MISSING VALUES & INITIAL ROWS")
print("="*60)

# Remove rows where lag features are 0 (first few rows with no history)
lag_columns = [col for col in df.columns if 'lag' in col.lower() or 'ret_mean' in col or 'ret_std' in col]
initial_invalid = (df[lag_columns] == 0).all(axis=1).sum()
print(f"Rows with all zero lag values: {initial_invalid}")

# Keep only rows where we have valid lag data
df_clean = df[~((df[lag_columns] == 0).all(axis=1))].copy()
print(f"Rows after removing invalid lags: {len(df_clean)}")

# ============================================
# STEP 2: REMOVE EXTREME OUTLIERS
# ============================================
print("\n" + "="*60)
print("STEP 2: DETECTING & REMOVING OUTLIERS")
print("="*60)

# Identify extreme market events (>3 standard deviations)
return_cols = ['return_pct', 'log_return', 'next_day_return']
for col in return_cols:
    mean_val = df_clean[col].mean()
    std_val = df_clean[col].std()
    outliers = ((df_clean[col] - mean_val).abs() > 3 * std_val).sum()
    print(f"{col}: {outliers} outliers (>{3*std_val:.4f})")

# Use Isolation Forest for multivariate outlier detection
numeric_features = df_clean.select_dtypes(include=[np.number]).columns
numeric_features = [col for col in numeric_features if col not in ['candidate', 'sentiment_label_enc']]

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(df_clean[numeric_features])
outlier_count = (outlier_labels == -1).sum()

print(f"\nIsolation Forest detected: {outlier_count} outliers ({outlier_count/len(df_clean)*100:.2f}%)")

# Create clean dataset
df_clean['is_outlier'] = outlier_labels
df_final = df_clean[df_clean['is_outlier'] == 1].drop('is_outlier', axis=1).copy()

print(f"Final dataset size: {len(df_final)} rows ({len(df_final)/len(df)*100:.1f}% of original)")

# ============================================
# STEP 3: FEATURE ENGINEERING
# ============================================
print("\n" + "="*60)
print("STEP 3: FEATURE ENGINEERING")
print("="*60)

# Convert sentiment label to categorical codes if needed
if df_final['sentiment_label'].dtype == 'object':
    df_final['sentiment_label_encoded'] = pd.Categorical(df_final['sentiment_label']).codes

# Create additional features
df_final['price_momentum'] = df_final['close'] - df_final['open']
df_final['volatility_ratio'] = df_final['high_low_spread'] / df_final['close']
df_final['sentiment_change'] = df_final['sentiment_score'] - df_final['sentiment_lag_1']

# Volume features
df_final['volume_ma_ratio'] = df_final['volume'] / df_final['volume'].rolling(window=5, min_periods=1).mean()

print("New features created:")
print("- price_momentum, volatility_ratio")
print("- sentiment_change, volume_ma_ratio")

# ============================================
# STEP 4: PREPARE FOR MODELING
# ============================================
print("\n" + "="*60)
print("STEP 4: FINAL DATASET PREPARATION")
print("="*60)

# Select relevant features for modeling
feature_columns = [
    'sentiment_score', 'sentiment_score_median', 'sentiment_count',
    'open', 'high', 'low', 'close', 'volume',
    'high_low_spread', 'open_close_gap',
    'ret_mean_3d', 'ret_std_3d', 'ret_mean_7d', 'ret_std_7d',
    'ret_mean_14d', 'ret_std_14d',
    'sentiment_lag_1', 'return_lag_1', 'sentiment_lag_2', 'return_lag_2',
    'sentiment_lag_3', 'return_lag_3',
    'sentiment_label_enc',
    'price_momentum', 'volatility_ratio', 'sentiment_change', 'volume_ma_ratio'
]

X = df_final[feature_columns].copy()
y_regression = df_final['next_day_return'].copy()
y_classification = df_final['next_day_dir'].copy()

# Handle any remaining NaN values
X = X.fillna(method='ffill').fillna(method='bfill')

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Target variable (regression) shape: {y_regression.shape}")
print(f"Target variable (classification) shape: {y_classification.shape}")
print(f"\nClass distribution:")
print(y_classification.value_counts(normalize=True))

# Save cleaned dataset
df_final.to_csv('nifty_sentiment_cleaned.csv', index=False)
print("\n✓ Cleaned dataset saved to 'nifty_sentiment_cleaned.csv'")

# ============================================
# MODEL RECOMMENDATIONS
# ============================================
print("\n" + "="*60)
print("RECOMMENDED MODELS FOR THIS DATASET")
print("="*60)

print("""
Based on your data characteristics:

📊 **BEST MODELS FOR REGRESSION (predicting next_day_return):**

1. **XGBoost Regressor** ⭐ HIGHLY RECOMMENDED
   - Pros: Handles non-linearity, feature interactions, robust to outliers
   - Handles time-series patterns well
   - Feature importance analysis built-in
   
2. **LightGBM Regressor** ⭐ HIGHLY RECOMMENDED
   - Pros: Faster than XGBoost, handles categorical features
   - Good for large datasets
   - Memory efficient
   
3. **Random Forest Regressor**
   - Pros: Robust, handles non-linearity
   - Less prone to overfitting than single decision trees
   
4. **LSTM (Long Short-Term Memory)**
   - Pros: Captures temporal dependencies
   - Good for sequential data
   - Cons: Requires more data, slower training

📈 **BEST MODELS FOR CLASSIFICATION (predicting next_day_dir):**

1. **XGBoost Classifier** ⭐ HIGHLY RECOMMENDED
   - Best overall performance for structured data
   
2. **LightGBM Classifier** ⭐ HIGHLY RECOMMENDED
   - Fast and accurate
   
3. **Random Forest Classifier**
   - Good baseline model
   
4. **Ensemble Models** (Stacking/Voting)
   - Combine multiple models for better performance

⚠️ **AVOID:**
- Simple Linear/Logistic Regression (data is non-linear)
- Basic Decision Trees (will overfit)
- SVM without proper kernel (too slow for this size)

🎯 **RECOMMENDATION:**
Start with **XGBoost** (both regressor and classifier) as it typically
performs best on financial time-series data with mixed features.
""")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("""
1. Load 'nifty_sentiment_cleaned.csv'
2. Split data chronologically (80% train, 20% test)
3. Scale features using RobustScaler
4. Train XGBoost model
5. Tune hyperparameters using cross-validation
6. Evaluate with RMSE, MAE (regression) or Accuracy, F1 (classification)
7. Analyze feature importance
""")