import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)

# ============================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================
print("="*70)
print("STEP 1: DATA LOADING AND CLEANING")
print("="*70)

df = pd.read_csv('preprocessed_nifty_sentiment.csv')
print(f"Original dataset shape: {df.shape}")

# Remove rows with invalid lag data
lag_columns = [col for col in df.columns if 'lag' in col.lower() or 'ret_mean' in col or 'ret_std' in col]
df_clean = df[~((df[lag_columns] == 0).all(axis=1))].copy()
print(f"After removing invalid lags: {df_clean.shape}")

# Less aggressive outlier removal
iso_forest = IsolationForest(contamination=0.03, random_state=42)
numeric_features = df_clean.select_dtypes(include=[np.number]).columns
numeric_features = [col for col in numeric_features if col not in ['candidate', 'sentiment_label_enc']]
outlier_labels = iso_forest.fit_predict(df_clean[numeric_features])
df_final = df_clean[outlier_labels == 1].copy()
print(f"After outlier removal: {df_final.shape}")

# Enhanced feature engineering
df_final['price_momentum'] = df_final['close'] - df_final['open']
df_final['volatility_ratio'] = df_final['high_low_spread'] / (df_final['close'] + 1e-8)
df_final['sentiment_change'] = df_final['sentiment_score'] - df_final['sentiment_lag_1']
df_final['volume_ma_ratio'] = df_final['volume'] / (df_final['volume'].rolling(window=5, min_periods=1).mean() + 1e-8)

# Additional technical indicators
df_final['price_range'] = (df_final['high'] - df_final['low']) / (df_final['close'] + 1e-8)
df_final['sentiment_momentum'] = df_final['sentiment_lag_1'] - df_final['sentiment_lag_2']
df_final['return_volatility'] = df_final[['return_lag_1', 'return_lag_2', 'return_lag_3']].std(axis=1)
df_final['sentiment_strength'] = df_final['sentiment_score'] * df_final['sentiment_count']

# Interaction features
df_final['sentiment_volume_interaction'] = df_final['sentiment_score'] * df_final['volume']
df_final['price_sentiment_interaction'] = df_final['price_momentum'] * df_final['sentiment_score']

# Convert date to datetime
df_final['date'] = pd.to_datetime(df_final['candidate'])
df_final = df_final.sort_values('date').reset_index(drop=True)

print(f"✓ Data cleaning completed. Final shape: {df_final.shape}")

# ============================================
# STEP 2: ENHANCED FEATURE SELECTION
# ============================================
print("\n" + "="*70)
print("STEP 2: ENHANCED FEATURE PREPARATION")
print("="*70)

feature_columns = [
    'sentiment_score', 'sentiment_score_median', 'sentiment_count',
    'open', 'high', 'low', 'close', 'volume',
    'high_low_spread', 'open_close_gap',
    'ret_mean_3d', 'ret_std_3d', 'ret_mean_7d', 'ret_std_7d',
    'ret_mean_14d', 'ret_std_14d',
    'sentiment_lag_1', 'return_lag_1', 'sentiment_lag_2', 'return_lag_2',
    'sentiment_lag_3', 'return_lag_3', 'sentiment_label_enc',
    'price_momentum', 'volatility_ratio', 'sentiment_change', 'volume_ma_ratio',
    'price_range', 'sentiment_momentum', 'return_volatility', 'sentiment_strength',
    'sentiment_volume_interaction', 'price_sentiment_interaction'
]

X = df_final[feature_columns].fillna(method='ffill').fillna(method='bfill')
y_regression = df_final['next_day_return'].values
y_classification = df_final['next_day_dir'].values
dates = df_final['date'].values
current_prices = df_final['close'].values

# Chronological train-test split (80-20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train_reg, y_test_reg = y_regression[:split_idx], y_regression[split_idx:]
y_train_clf, y_test_clf = y_classification[:split_idx], y_classification[split_idx:]
dates_train, dates_test = dates[:split_idx], dates[split_idx:]
prices_train, prices_test = current_prices[:split_idx], current_prices[split_idx:]

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Class distribution (train): Down={sum(y_train_clf==0)}, Up={sum(y_train_clf==1)} ({sum(y_train_clf==1)/len(y_train_clf)*100:.1f}% Up)")
print(f"Class distribution (test): Down={sum(y_test_clf==0)}, Up={sum(y_test_clf==1)} ({sum(y_test_clf==1)/len(y_test_clf)*100:.1f}% Up)")

# Feature selection for classification
selector = SelectKBest(mutual_info_classif, k=min(20, len(feature_columns)))
X_train_selected = selector.fit_transform(X_train, y_train_clf)
X_test_selected = selector.transform(X_test)

selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
print(f"\nTop features selected: {len(selected_features)}")
print("Selected features:", selected_features[:10])

# Scale features
scaler_standard = StandardScaler()
X_train_scaled = scaler_standard.fit_transform(X_train_selected)
X_test_scaled = scaler_standard.transform(X_test_selected)

# ============================================
# STEP 3: ENHANCED XGBOOST MODELS
# ============================================
print("\n" + "="*70)
print("STEP 3: TRAINING ENHANCED XGBOOST MODELS")
print("="*70)

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = sum(y_train_clf == 0) / sum(y_train_clf == 1)
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# XGBoost Regressor (for return prediction)
print("\n[1/5] XGBoost Regressor (Optimized)...")
xgb_reg_optimized = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)
xgb_reg_optimized.fit(X_train_scaled, y_train_reg)
xgb_pred_reg = xgb_reg_optimized.predict(X_test_scaled)

# XGBoost Classifier (for direction prediction)
print("[2/5] XGBoost Classifier (Optimized)...")
xgb_clf_optimized = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbosity=0,
    eval_metric='logloss'
)
xgb_clf_optimized.fit(X_train_scaled, y_train_clf)
xgb_pred_clf_opt = xgb_clf_optimized.predict(X_test_scaled)
xgb_pred_proba_opt = xgb_clf_optimized.predict_proba(X_test_scaled)[:, 1]

# Random Forest Classifier
print("[3/5] Random Forest Classifier...")
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_clf.fit(X_train_scaled, y_train_clf)
rf_pred_clf = rf_clf.predict(X_test_scaled)
rf_pred_proba = rf_clf.predict_proba(X_test_scaled)[:, 1]

# Gradient Boosting Classifier
print("[4/5] Gradient Boosting Classifier...")
gb_clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    subsample=0.85,
    random_state=42
)
gb_clf.fit(X_train_scaled, y_train_clf)
gb_pred_clf = gb_clf.predict(X_test_scaled)
gb_pred_proba = gb_clf.predict_proba(X_test_scaled)[:, 1]

# Ensemble prediction (voting)
print("[5/5] Creating Ensemble...")
ensemble_proba = (xgb_pred_proba_opt + rf_pred_proba + gb_pred_proba) / 3
ensemble_pred = (ensemble_proba > 0.5).astype(int)

print("✓ Enhanced models trained successfully")

# ============================================
# STEP 4: ENHANCED LSTM/GRU MODELS
# ============================================
print("\n" + "="*70)
print("STEP 4: TRAINING ENHANCED LSTM/GRU MODELS")
print("="*70)

lookback = 5

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq_clf = create_sequences(X_train_scaled, y_train_clf, lookback)
X_test_seq, y_test_seq_clf = create_sequences(X_test_scaled, y_test_clf, lookback)

print(f"Sequence shape: {X_train_seq.shape}")

# Enhanced LSTM Classifier
print("\n[1/2] Training Enhanced Bidirectional LSTM Classifier...")
lstm_clf_enhanced = Sequential([
    Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01)), 
                  input_shape=(lookback, X_train_scaled.shape[1])),
    Dropout(0.3),
    Bidirectional(LSTM(16, return_sequences=False, kernel_regularizer=l2(0.01))),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.001)
lstm_clf_enhanced.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001)

lstm_clf_enhanced.fit(
    X_train_seq, y_train_seq_clf, 
    epochs=100, 
    batch_size=16,
    validation_split=0.2, 
    callbacks=[early_stop, reduce_lr], 
    verbose=0,
    class_weight={0: scale_pos_weight, 1: 1.0}
)
lstm_pred_clf_enh = (lstm_clf_enhanced.predict(X_test_seq, verbose=0).flatten() > 0.5).astype(int)

# Enhanced GRU Classifier
print("[2/2] Training Enhanced Bidirectional GRU Classifier...")
gru_clf_enhanced = Sequential([
    Bidirectional(GRU(32, return_sequences=True, kernel_regularizer=l2(0.01)), 
                  input_shape=(lookback, X_train_scaled.shape[1])),
    Dropout(0.3),
    Bidirectional(GRU(16, return_sequences=False, kernel_regularizer=l2(0.01))),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

gru_clf_enhanced.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
gru_clf_enhanced.fit(
    X_train_seq, y_train_seq_clf, 
    epochs=100, 
    batch_size=16,
    validation_split=0.2, 
    callbacks=[early_stop, reduce_lr], 
    verbose=0,
    class_weight={0: scale_pos_weight, 1: 1.0}
)
gru_pred_clf_enh = (gru_clf_enhanced.predict(X_test_seq, verbose=0).flatten() > 0.5).astype(int)

print("✓ Enhanced LSTM/GRU models trained successfully")

# ============================================
# STEP 5: PRICE PREDICTIONS
# ============================================
print("\n" + "="*70)
print("STEP 5: CALCULATING PRICE PREDICTIONS")
print("="*70)

# Calculate predicted next day prices using returns
predicted_prices_xgb_reg = prices_test * (1 + xgb_pred_reg)
predicted_prices_xgb_clf = prices_test * (1 + np.where(xgb_pred_clf_opt == 1, abs(xgb_pred_reg), -abs(xgb_pred_reg)))
predicted_prices_ensemble = prices_test * (1 + np.where(ensemble_pred == 1, abs(xgb_pred_reg), -abs(xgb_pred_reg)))

# Actual next day prices
actual_next_prices = prices_test * (1 + y_test_reg)

print(f"Price prediction range:")
print(f"  Current prices: {prices_test.min():.2f} to {prices_test.max():.2f}")
print(f"  Actual next day: {actual_next_prices.min():.2f} to {actual_next_prices.max():.2f}")
print(f"  Predicted (XGB Reg): {predicted_prices_xgb_reg.min():.2f} to {predicted_prices_xgb_reg.max():.2f}")
print(f"  Predicted (Ensemble): {predicted_prices_ensemble.min():.2f} to {predicted_prices_ensemble.max():.2f}")

# Calculate price prediction errors
price_error_xgb_reg = np.abs(predicted_prices_xgb_reg - actual_next_prices)
price_error_ensemble = np.abs(predicted_prices_ensemble - actual_next_prices)

print(f"\nMean Absolute Price Error:")
print(f"  XGB Regressor: {price_error_xgb_reg.mean():.2f}")
print(f"  Ensemble: {price_error_ensemble.mean():.2f}")

# ============================================
# STEP 6: COMPREHENSIVE EVALUATION
# ============================================
print("\n" + "="*70)
print("STEP 6: COMPREHENSIVE MODEL EVALUATION")
print("="*70)

def calculate_detailed_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")
    print(f"  🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Down (0)', 'Up (1)'], digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Down  Up")
    print(f"  Actual Down    {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"         Up      {cm[1,0]:3d}  {cm[1,1]:3d}")
    
    return {'Accuracy': acc}

print("\n📈 CLASSIFICATION METRICS (Predicting Direction: Up/Down)")

metrics_clf = {}
metrics_clf['XGBoost (Optimized)'] = calculate_detailed_metrics(y_test_clf, xgb_pred_clf_opt, "XGBoost Classifier (Optimized)")
metrics_clf['Random Forest'] = calculate_detailed_metrics(y_test_clf, rf_pred_clf, "Random Forest Classifier")
metrics_clf['Gradient Boosting'] = calculate_detailed_metrics(y_test_clf, gb_pred_clf, "Gradient Boosting Classifier")
metrics_clf['Ensemble (Voting)'] = calculate_detailed_metrics(y_test_clf, ensemble_pred, "Ensemble Model (XGB+RF+GB)")

y_test_clf_seq = y_test_clf[lookback:]
metrics_clf['LSTM (Enhanced)'] = calculate_detailed_metrics(y_test_clf_seq, lstm_pred_clf_enh, "Enhanced Bidirectional LSTM")
metrics_clf['GRU (Enhanced)'] = calculate_detailed_metrics(y_test_clf_seq, gru_pred_clf_enh, "Enhanced Bidirectional GRU")

# ============================================
# STEP 7: SAVE MODELS
# ============================================
print("\n" + "="*70)
print("STEP 7: SAVING MODELS AND ARTIFACTS")
print("="*70)

# Create directories if they don't exist
import os
os.makedirs('models', exist_ok=True)
os.makedirs('predictions', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
print("✓ Created necessary directories")

# Save all models
joblib.dump(xgb_reg_optimized, 'models/xgb_regressor.pkl')
joblib.dump(xgb_clf_optimized, 'models/xgb_classifier.pkl')
joblib.dump(rf_clf, 'models/random_forest_classifier.pkl')
joblib.dump(gb_clf, 'models/gradient_boosting_classifier.pkl')
joblib.dump(scaler_standard, 'models/scaler.pkl')
joblib.dump(selector, 'models/feature_selector.pkl')
joblib.dump(selected_features, 'models/selected_features.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl')

# Save deep learning models
lstm_clf_enhanced.save('models/lstm_classifier.h5')
gru_clf_enhanced.save('models/gru_classifier.h5')

print("✓ Saved xgb_regressor.pkl")
print("✓ Saved xgb_classifier.pkl")
print("✓ Saved random_forest_classifier.pkl")
print("✓ Saved gradient_boosting_classifier.pkl")
print("✓ Saved lstm_classifier.h5")
print("✓ Saved gru_classifier.h5")
print("✓ Saved scaler.pkl")
print("✓ Saved feature_selector.pkl")
print("✓ Saved selected_features.pkl")
print("✓ Saved feature_columns.pkl")

# ============================================
# STEP 8: SAVE PREDICTIONS AND COMPARISONS
# ============================================
print("\n" + "="*70)
print("STEP 8: SAVING PREDICTIONS AND COMPARISONS")
print("="*70)

# Create comprehensive predictions dataframe
predictions_df = pd.DataFrame({
    'date': dates_test,
    'current_price': prices_test,
    'actual_next_price': actual_next_prices,
    'actual_return': y_test_reg,
    'actual_direction': y_test_clf,
    
    # XGBoost Regressor predictions
    'pred_return_xgb_reg': xgb_pred_reg,
    'pred_price_xgb_reg': predicted_prices_xgb_reg,
    'price_error_xgb_reg': price_error_xgb_reg,
    
    # XGBoost Classifier predictions
    'pred_direction_xgb_clf': xgb_pred_clf_opt,
    'pred_proba_xgb_clf': xgb_pred_proba_opt,
    'direction_correct_xgb_clf': (xgb_pred_clf_opt == y_test_clf).astype(int),
    
    # Ensemble predictions
    'pred_direction_ensemble': ensemble_pred,
    'pred_proba_ensemble': ensemble_proba,
    'pred_price_ensemble': predicted_prices_ensemble,
    'price_error_ensemble': price_error_ensemble,
    'direction_correct_ensemble': (ensemble_pred == y_test_clf).astype(int),
})

predictions_df.to_csv('predictions/model_predictions_comparison.csv', index=False)
print("✓ Saved predictions/model_predictions_comparison.csv")

# Create summary statistics
summary_stats = pd.DataFrame({
    'Model': ['XGBoost Regressor', 'XGBoost Classifier', 'Ensemble'],
    'Mean_Price_Error': [
        price_error_xgb_reg.mean(),
        np.nan,
        price_error_ensemble.mean()
    ],
    'RMSE_Price': [
        np.sqrt(mean_squared_error(actual_next_prices, predicted_prices_xgb_reg)),
        np.nan,
        np.sqrt(mean_squared_error(actual_next_prices, predicted_prices_ensemble))
    ],
    'Direction_Accuracy': [
        np.nan,
        accuracy_score(y_test_clf, xgb_pred_clf_opt),
        accuracy_score(y_test_clf, ensemble_pred)
    ]
})
summary_stats.to_csv('predictions/model_summary_statistics.csv', index=False)
print("✓ Saved predictions/model_summary_statistics.csv")

print("\nFirst 10 predictions:")
print(predictions_df[['date', 'current_price', 'actual_next_price', 'pred_price_ensemble', 
                      'actual_direction', 'pred_direction_ensemble']].head(10))

# ============================================
# STEP 9: VISUALIZATIONS
# ============================================
print("\n" + "="*70)
print("STEP 9: CREATING COMPREHENSIVE VISUALIZATIONS")
print("="*70)

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# Plot 1: Model Accuracy Comparison
ax1 = fig.add_subplot(gs[0, :])
model_names = list(metrics_clf.keys())
acc_values = [metrics_clf[m]['Accuracy'] * 100 for m in model_names]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
bars = ax1.bar(range(len(model_names)), acc_values, color=colors[:len(model_names)])
ax1.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% Target')
ax1.set_title('Model Accuracy Comparison - Target: 80%+', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=15, ha='right')
ax1.set_ylim([min(acc_values) - 5, 100])
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend()
for i, (v, name) in enumerate(zip(acc_values, model_names)):
    color = 'green' if v >= 80 else 'red'
    ax1.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold', color=color)

# Plot 2: Actual vs Predicted Prices (XGBoost Regressor)
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(dates_test, actual_next_prices, label='Actual Next Day Price', color='#2c3e50', linewidth=2)
ax2.plot(dates_test, predicted_prices_xgb_reg, label='Predicted Price (XGB Reg)', color='#e74c3c', linewidth=1.5, alpha=0.7)
ax2.set_title('Price Prediction - XGBoost Regressor', fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Actual vs Predicted Prices (Ensemble)
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(dates_test, actual_next_prices, label='Actual Next Day Price', color='#2c3e50', linewidth=2)
ax3.plot(dates_test, predicted_prices_ensemble, label='Predicted Price (Ensemble)', color='#27ae60', linewidth=1.5, alpha=0.7)
ax3.set_title('Price Prediction - Ensemble Model', fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Price Prediction Errors
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist([price_error_xgb_reg, price_error_ensemble], 
         bins=30, alpha=0.6, label=['XGB Regressor', 'Ensemble'])
ax4.set_title('Price Prediction Error Distribution', fontweight='bold')
ax4.set_xlabel('Absolute Price Error')
ax4.set_ylabel('Frequency')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Confusion Matrix - Best Model
best_model = max(metrics_clf, key=lambda x: metrics_clf[x]['Accuracy'])
best_acc = metrics_clf[best_model]['Accuracy']

if best_model in ['LSTM (Enhanced)', 'GRU (Enhanced)']:
    y_true_best = y_test_clf_seq
    if best_model == 'LSTM (Enhanced)':
        y_pred_best = lstm_pred_clf_enh
    else:
        y_pred_best = gru_pred_clf_enh
else:
    y_true_best = y_test_clf
    if best_model == 'XGBoost (Optimized)':
        y_pred_best = xgb_pred_clf_opt
    elif best_model == 'Random Forest':
        y_pred_best = rf_pred_clf
    elif best_model == 'Gradient Boosting':
        y_pred_best = gb_pred_clf
    else:
        y_pred_best = ensemble_pred

cm = confusion_matrix(y_true_best, y_pred_best)
ax5 = fig.add_subplot(gs[2, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5, cbar=False)
ax5.set_title(f'Confusion Matrix - {best_model}\nAccuracy: {best_acc*100:.2f}%', fontweight='bold')
ax5.set_ylabel('Actual')
ax5.set_xlabel('Predicted')
ax5.set_xticklabels(['Down', 'Up'])
ax5.set_yticklabels(['Down', 'Up'])

# Plot 6: Feature Importance
ax6 = fig.add_subplot(gs[2, 1])
feature_importance = xgb_clf_optimized.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:]
top_features_names = [selected_features[i] if i < len(selected_features) else f'Feature_{i}' for i in top_features_idx]
ax6.barh(range(10), feature_importance[top_features_idx], color='#3498db')
ax6.set_yticks(range(10))
ax6.set_yticklabels(top_features_names, fontsize=9)
ax6.set_xlabel('Importance')
ax6.set_title('Top 10 Feature Importances', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

# Plot 7: Prediction Probability Distribution
ax7 = fig.add_subplot(gs[2, 2])
ax7.hist(ensemble_proba, bins=30, alpha=0.7, color='#9b59b6')
ax7.set_title('Ensemble Prediction Probability\nDistribution', fontweight='bold')
ax7.set_xlabel('Probability of Up Movement')
ax7.set_ylabel('Frequency')
ax7.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Price Error Over Time (XGBoost Regressor)
ax8 = fig.add_subplot(gs[3, 0])
ax8.plot(dates_test, price_error_xgb_reg, color='#e74c3c', linewidth=1.5, alpha=0.7)
ax8.set_title('Price Prediction Error Over Time\n(XGBoost Regressor)', fontweight='bold')
ax8.set_xlabel('Date')
ax8.set_ylabel('Absolute Error')
ax8.axhline(y=price_error_xgb_reg.mean(), color='blue', linestyle='--', 
            linewidth=2, label=f'Mean Error: {price_error_xgb_reg.mean():.2f}')
ax8.legend()
ax8.grid(True, alpha=0.3)
ax8.tick_params(axis='x', rotation=45)

# Plot 9: Price Error Over Time (Ensemble)
ax9 = fig.add_subplot(gs[3, 1])
ax9.plot(dates_test, price_error_ensemble, color='#27ae60', linewidth=1.5, alpha=0.7)
ax9.set_title('Price Prediction Error Over Time\n(Ensemble)', fontweight='bold')
ax9.set_xlabel('Date')
ax9.set_ylabel('Absolute Error')
ax9.axhline(y=price_error_ensemble.mean(), color='blue', linestyle='--', 
            linewidth=2, label=f'Mean Error: {price_error_ensemble.mean():.2f}')
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.tick_params(axis='x', rotation=45)

# Plot 10: Model Summary Statistics
ax10 = fig.add_subplot(gs[3, 2])
ax10.axis('off')
summary_text = f"""
MODEL PERFORMANCE SUMMARY
{'='*35}

BEST CLASSIFICATION MODEL
{best_model}
Accuracy: {best_acc*100:.2f}%

PRICE PREDICTION METRICS
{'─'*35}
XGBoost Regressor:
  Mean Abs Error: {price_error_xgb_reg.mean():.2f}
  RMSE: {np.sqrt(mean_squared_error(actual_next_prices, predicted_prices_xgb_reg)):.2f}

Ensemble:
  Mean Abs Error: {price_error_ensemble.mean():.2f}
  RMSE: {np.sqrt(mean_squared_error(actual_next_prices, predicted_prices_ensemble)):.2f}

ALL CLASSIFICATION MODELS
{'─'*35}
"""
for name, metrics in sorted(metrics_clf.items(), key=lambda x: x[1]['Accuracy'], reverse=True):
    summary_text += f"{name}: {metrics['Accuracy']*100:.2f}%\n"

ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, 
         fontfamily='monospace', fontsize=8, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Comprehensive Model Evaluation - NIFTY Sentiment Analysis with Price Predictions', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('visualizations/comprehensive_model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualizations/comprehensive_model_evaluation.png")

# Create separate detailed price comparison plot
fig2, axes = plt.subplots(2, 1, figsize=(16, 10))

# Detailed price comparison - first half of test set
mid_point = len(dates_test) // 2
axes[0].plot(dates_test[:mid_point], actual_next_prices[:mid_point], 
            label='Actual Price', color='#2c3e50', linewidth=2, marker='o', markersize=4)
axes[0].plot(dates_test[:mid_point], predicted_prices_xgb_reg[:mid_point], 
            label='XGBoost Regressor', color='#e74c3c', linewidth=1.5, marker='s', markersize=3, alpha=0.7)
axes[0].plot(dates_test[:mid_point], predicted_prices_ensemble[:mid_point], 
            label='Ensemble', color='#27ae60', linewidth=1.5, marker='^', markersize=3, alpha=0.7)
axes[0].set_title('Actual vs Predicted Prices - First Half of Test Period', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Detailed price comparison - second half of test set
axes[1].plot(dates_test[mid_point:], actual_next_prices[mid_point:], 
            label='Actual Price', color='#2c3e50', linewidth=2, marker='o', markersize=4)
axes[1].plot(dates_test[mid_point:], predicted_prices_xgb_reg[mid_point:], 
            label='XGBoost Regressor', color='#e74c3c', linewidth=1.5, marker='s', markersize=3, alpha=0.7)
axes[1].plot(dates_test[mid_point:], predicted_prices_ensemble[mid_point:], 
            label='Ensemble', color='#27ae60', linewidth=1.5, marker='^', markersize=3, alpha=0.7)
axes[1].set_title('Actual vs Predicted Prices - Second Half of Test Period', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Price')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/detailed_price_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualizations/detailed_price_predictions.png")

# Create accuracy over time plot
fig3, ax = plt.subplots(figsize=(14, 6))
window_size = 10
rolling_accuracy = pd.Series(predictions_df['direction_correct_ensemble']).rolling(window=window_size).mean() * 100
ax.plot(dates_test, rolling_accuracy, color='#3498db', linewidth=2)
ax.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% Target')
ax.axhline(y=rolling_accuracy.mean(), color='green', linestyle='--', 
          linewidth=2, label=f'Mean: {rolling_accuracy.mean():.2f}%')
ax.set_title(f'Rolling Accuracy Over Time (Window: {window_size} days)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Accuracy (%)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('visualizations/rolling_accuracy_over_time.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualizations/rolling_accuracy_over_time.png")

# ============================================
# STEP 10: CREATE PREDICTION FUNCTION
# ============================================
print("\n" + "="*70)
print("STEP 10: CREATING PREDICTION FUNCTION")
print("="*70)

def predict_next_day(input_data_dict):
    """
    Make predictions for next day price and direction
    
    Parameters:
    -----------
    input_data_dict : dict
        Dictionary containing all required features
        
    Returns:
    --------
    dict : Predictions including price, direction, and probabilities
    """
    # Load models and artifacts
    xgb_reg = joblib.load('models/xgb_regressor.pkl')
    xgb_clf = joblib.load('models/xgb_classifier.pkl')
    rf_clf = joblib.load('models/random_forest_classifier.pkl')
    gb_clf = joblib.load('models/gradient_boosting_classifier.pkl')
    scaler = joblib.load('models/scaler.pkl')
    selector = joblib.load('models/feature_selector.pkl')
    feature_cols = joblib.load('models/feature_columns.pkl')
    
    # Create dataframe from input
    input_df = pd.DataFrame([input_data_dict])
    
    # Ensure all features are present
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Select and scale features
    X_input = input_df[feature_cols]
    X_selected = selector.transform(X_input)
    X_scaled = scaler.transform(X_selected)
    
    # Make predictions
    pred_return = xgb_reg.predict(X_scaled)[0]
    pred_direction = xgb_clf.predict(X_scaled)[0]
    pred_proba = xgb_clf.predict_proba(X_scaled)[0, 1]
    
    # Ensemble predictions
    xgb_proba = xgb_clf.predict_proba(X_scaled)[0, 1]
    rf_proba = rf_clf.predict_proba(X_scaled)[0, 1]
    gb_proba = gb_clf.predict_proba(X_scaled)[0, 1]
    ensemble_proba = (xgb_proba + rf_proba + gb_proba) / 3
    ensemble_direction = 1 if ensemble_proba > 0.5 else 0
    
    # Calculate predicted price
    current_price = input_data_dict.get('close', 0)
    pred_price_reg = current_price * (1 + pred_return)
    pred_price_ensemble = current_price * (1 + (abs(pred_return) if ensemble_direction == 1 else -abs(pred_return)))
    
    return {
        'current_price': current_price,
        'predicted_return': pred_return,
        'predicted_price_regressor': pred_price_reg,
        'predicted_price_ensemble': pred_price_ensemble,
        'predicted_direction_xgb': 'Up' if pred_direction == 1 else 'Down',
        'predicted_direction_ensemble': 'Up' if ensemble_direction == 1 else 'Down',
        'probability_up_xgb': pred_proba,
        'probability_up_ensemble': ensemble_proba,
        'confidence': 'High' if abs(ensemble_proba - 0.5) > 0.2 else 'Medium' if abs(ensemble_proba - 0.5) > 0.1 else 'Low'
    }

# Save the prediction function
import dill
with open('models/prediction_function.pkl', 'wb') as f:
    dill.dump(predict_next_day, f)

print("✓ Created and saved prediction function")

# Test the prediction function with last test sample
test_input = X_test.iloc[-1].to_dict()
test_prediction = predict_next_day(test_input)

print("\n" + "="*70)
print("EXAMPLE PREDICTION (Last Test Sample)")
print("="*70)
for key, value in test_prediction.items():
    print(f"  {key}: {value}")

# ============================================
# STEP 11: FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("📋 FINAL SUMMARY AND RECOMMENDATIONS")
print("="*70)

best_model = max(metrics_clf, key=lambda x: metrics_clf[x]['Accuracy'])
best_acc = metrics_clf[best_model]['Accuracy']

print(f"\n🏆 BEST CLASSIFICATION MODEL: {best_model}")
print(f"   Accuracy: {best_acc*100:.2f}%")

if best_acc >= 0.80:
    print(f"\n✅ SUCCESS! Target of 80% accuracy ACHIEVED!")
else:
    print(f"\n⚠️  Current best: {best_acc*100:.2f}% (Target: 80%)")
    print(f"   Gap to target: {(0.80 - best_acc)*100:.2f}%")

print(f"\n💰 PRICE PREDICTION PERFORMANCE:")
print(f"   XGBoost Regressor:")
print(f"     - Mean Absolute Error: {price_error_xgb_reg.mean():.2f}")
print(f"     - RMSE: {np.sqrt(mean_squared_error(actual_next_prices, predicted_prices_xgb_reg)):.2f}")
print(f"   Ensemble:")
print(f"     - Mean Absolute Error: {price_error_ensemble.mean():.2f}")
print(f"     - RMSE: {np.sqrt(mean_squared_error(actual_next_prices, predicted_prices_ensemble)):.2f}")

print("\n📊 ALL MODEL ACCURACIES:")
for name, metrics in sorted(metrics_clf.items(), key=lambda x: x[1]['Accuracy'], reverse=True):
    status = "✅" if metrics['Accuracy'] >= 0.80 else "❌"
    print(f"   {status} {name}: {metrics['Accuracy']*100:.2f}%")

print("\n💡 KEY IMPROVEMENTS MADE:")
print("   ✓ Enhanced feature engineering (interaction terms)")
print("   ✓ Feature selection using mutual information")
print("   ✓ Optimized hyperparameters for all models")
print("   ✓ Class imbalance handling (scale_pos_weight)")
print("   ✓ Ensemble methods (voting)")
print("   ✓ Bidirectional LSTM/GRU with regularization")
print("   ✓ Reduced lookback period to prevent overfitting")
print("   ✓ Price prediction capabilities added")

print("\n📦 SAVED ARTIFACTS:")
print("   Models:")
print("     - models/xgb_regressor.pkl")
print("     - models/xgb_classifier.pkl")
print("     - models/random_forest_classifier.pkl")
print("     - models/gradient_boosting_classifier.pkl")
print("     - models/lstm_classifier.h5")
print("     - models/gru_classifier.h5")
print("     - models/scaler.pkl")
print("     - models/feature_selector.pkl")
print("     - models/prediction_function.pkl")
print("   ")
print("   Predictions:")
print("     - predictions/model_predictions_comparison.csv")
print("     - predictions/model_summary_statistics.csv")
print("   ")
print("   Visualizations:")
print("     - visualizations/comprehensive_model_evaluation.png")
print("     - visualizations/detailed_price_predictions.png")
print("     - visualizations/rolling_accuracy_over_time.png")

print("\n🎯 HOW TO USE THE MODELS:")
print("   1. Load the prediction function:")
print("      import joblib")
print("      import dill")
print("      with open('models/prediction_function.pkl', 'rb') as f:")
print("          predict_next_day = dill.load(f)")
print("   ")
print("   2. Prepare input data (dictionary with all features)")
print("   ")
print("   3. Make prediction:")
print("      prediction = predict_next_day(input_data)")
print("   ")
print("   4. Use the results:")
print("      - prediction['predicted_price_ensemble']")
print("      - prediction['predicted_direction_ensemble']")
print("      - prediction['confidence']")

print("\n🎯 RECOMMENDATIONS FOR FURTHER IMPROVEMENT:")
print("   1. Collect more training data for better generalization")
print("   2. Implement walk-forward validation for time series")
print("   3. Add more external features (VIX, global indices, etc.)")
print("   4. Try stacking/blending ensemble methods")
print("   5. Experiment with different threshold values")
print("   6. Consider deep learning architectures (Transformers)")
print("   7. Implement risk management strategies")
print("   8. Backtest trading strategies based on predictions")

print("\n✓ Enhanced analysis with price predictions completed successfully!")
print("="*70)