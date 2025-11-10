import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('C:/Users/madha/OneDrive/Desktop/NLP/preprocessed_nifty_sentiment.csv')

# Select only OHLC columns and target
ohlc_columns = ['open', 'high', 'low', 'close', 'volume']
df_ohlc = df[ohlc_columns + ['next_day_dir']].copy()

# Remove any rows with missing values
df_ohlc = df_ohlc.dropna()

print(f"Dataset shape: {df_ohlc.shape}")
print(f"\nTarget distribution:\n{df_ohlc['next_day_dir'].value_counts()}")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Create technical indicators
df_ohlc['high_low_range'] = df_ohlc['high'] - df_ohlc['low']
df_ohlc['open_close_diff'] = df_ohlc['close'] - df_ohlc['open']
df_ohlc['price_momentum'] = df_ohlc['close'].pct_change()

# Moving averages
for window in [3, 5, 7]:
    df_ohlc[f'ma_{window}'] = df_ohlc['close'].rolling(window=window).mean()
    df_ohlc[f'vol_ma_{window}'] = df_ohlc['volume'].rolling(window=window).mean()

# Volatility
df_ohlc['volatility_7'] = df_ohlc['close'].rolling(window=7).std()

# Remove NaN values created by rolling windows
df_ohlc = df_ohlc.dropna()

print(f"\nDataset shape after feature engineering: {df_ohlc.shape}")

# =============================================================================
# PREPARE DATA FOR MODELS
# =============================================================================

# Separate features and target
X = df_ohlc.drop('next_day_dir', axis=1)
y = df_ohlc['next_day_dir']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# =============================================================================
# XGBOOST MODEL
# =============================================================================

print("\n" + "="*80)
print("TRAINING XGBOOST MODEL")
print("="*80)

# Scale features for XGBoost (optional but can help)
scaler_xgb = MinMaxScaler()
X_train_scaled = scaler_xgb.fit_transform(X_train)
X_test_scaled = scaler_xgb.transform(X_test)

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

# Predictions
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluate XGBoost
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"\nXGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_xgb)}")

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('xgb_confusion_matrix.png')
plt.close()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Important Features:\n{feature_importance.head(10)}")

# Save XGBoost model
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(scaler_xgb, 'xgb_scaler.pkl')
print("\n✓ XGBoost model saved as 'xgboost_model.pkl'")
print("✓ XGBoost scaler saved as 'xgb_scaler.pkl'")

# =============================================================================
# LSTM MODEL
# =============================================================================

print("\n" + "="*80)
print("TRAINING LSTM MODEL")
print("="*80)

# Scale data for LSTM (LSTM is sensitive to scale)
scaler_lstm = MinMaxScaler()
X_train_lstm = scaler_lstm.fit_transform(X_train)
X_test_lstm = scaler_lstm.transform(X_test)

# Reshape for LSTM (samples, timesteps, features)
# Using a lookback window of 10 days
lookback = 10

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        ys.append(y.iloc[i + lookback])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_lstm, y_train, lookback)
X_test_seq, y_test_seq = create_sequences(X_test_lstm, y_test, lookback)

print(f"\nLSTM Train shape: {X_train_seq.shape}")
print(f"LSTM Test shape: {X_test_seq.shape}")

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nLSTM Model Architecture:")
lstm_model.summary()

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train LSTM
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Predictions
y_pred_lstm_proba = lstm_model.predict(X_test_seq)
y_pred_lstm = (y_pred_lstm_proba > 0.5).astype(int).flatten()

# Evaluate LSTM
lstm_accuracy = accuracy_score(y_test_seq, y_pred_lstm)
print(f"\nLSTM Accuracy: {lstm_accuracy:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test_seq, y_pred_lstm)}")

# Confusion Matrix
cm_lstm = confusion_matrix(y_test_seq, y_pred_lstm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Greens')
plt.title('LSTM Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('lstm_confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('lstm_training_history.png')
plt.close()

# Save LSTM model
lstm_model.save('lstm_model.h5')
joblib.dump(scaler_lstm, 'lstm_scaler.pkl')
print("\n✓ LSTM model saved as 'lstm_model.h5'")
print("✓ LSTM scaler saved as 'lstm_scaler.pkl'")

# =============================================================================
# FINAL COMPARISON
# =============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"LSTM Accuracy:    {lstm_accuracy:.4f}")

# Create comparison plot
models = ['XGBoost', 'LSTM']
accuracies = [xgb_accuracy, lstm_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['#1f77b4', '#2ca02c'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim([0, 1])
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
plt.savefig('model_comparison.png')
plt.close()

print("\n" + "="*80)
print("ALL MODELS AND SCALERS SAVED SUCCESSFULLY!")
print("="*80)
print("\nSaved files:")
print("1. xgboost_model.pkl")
print("2. xgb_scaler.pkl")
print("3. lstm_model.h5")
print("4. lstm_scaler.pkl")
print("\nGenerated plots:")
print("1. xgb_confusion_matrix.png")
print("2. lstm_confusion_matrix.png")
print("3. lstm_training_history.png")
print("4. model_comparison.png")

# =============================================================================
# LOADING MODELS (FOR FUTURE PREDICTIONS)
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE: HOW TO LOAD AND USE SAVED MODELS")
print("="*80)

print("""
# Load XGBoost Model:
loaded_xgb = joblib.load('xgboost_model.pkl')
loaded_xgb_scaler = joblib.load('xgb_scaler.pkl')

# Make predictions:
new_data_scaled = loaded_xgb_scaler.transform(new_data)
predictions = loaded_xgb.predict(new_data_scaled)

# Load LSTM Model:
from tensorflow import keras
loaded_lstm = keras.models.load_model('lstm_model.h5')
loaded_lstm_scaler = joblib.load('lstm_scaler.pkl')

# Make predictions (remember to create sequences):
new_data_scaled = loaded_lstm_scaler.transform(new_data)
new_data_seq = create_sequences(new_data_scaled, lookback=10)
predictions = loaded_lstm.predict(new_data_seq)
""")