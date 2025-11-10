# lstm_with_and_without_attention_fixed.py
# Requires: pandas, numpy, scikit-learn, tensorflow, matplotlib
# Optional: imbalanced-learn (for oversampling)

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, balanced_accuracy_score, matthews_corrcoef
)
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import matplotlib.pyplot as plt

# reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ------------- USER PARAMETERS -------------
CSV_PATH = r"C:/Users/madha/OneDrive/Desktop/NLP/lstm/preprocessed_nifty_sentiment.csv"
DATE_COL = None
LOOKBACK = 14            # increased from 7 -> 14 for more context
TEST_SIZE = 0.20
VAL_SIZE = 0.12
BATCH_SIZE = 32
EPOCHS = 80
MODEL_DIR = "models_lstm_compare_fixed"
os.makedirs(MODEL_DIR, exist_ok=True)

# Toggle strategies
USE_OVERSAMPLING = False   # set True if you installed imbalanced-learn
USE_FOCAL_LOSS = False     # set True to use focal loss
# -------------------------------------------

# ------------- helper attention layer -------------
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1],), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1],), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, inputs, mask=None):
        scores = K.dot(inputs, K.expand_dims(self.W))
        scores = K.squeeze(scores, axis=-1)
        scores = scores + self.b
        alpha = K.softmax(scores)
        alpha_expanded = K.expand_dims(alpha)
        weighted = inputs * alpha_expanded
        context = K.sum(weighted, axis=1)
        return context
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

# ------------- focal loss (optional) -------------
def focal_loss(alpha=0.5, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)
    return loss

# ------------- load & preprocess -------------
print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

if DATE_COL and DATE_COL in df.columns:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

default_features = [
    "sentiment_score", "sentiment_score_median", "sentiment_count", "sentiment_label_enc",
    "open","high","low","close","volume",
    "return_lag_1","return_lag_2","return_lag_3",
    "sentiment_lag_1","sentiment_lag_2","sentiment_lag_3",
    "ret_mean_3d","ret_std_3d","ret_mean_7d","ret_std_7d"
]
features = [c for c in default_features if c in df.columns]
if len(features) == 0:
    raise RuntimeError("None of the expected features exist in the CSV. Update default_features list.")
print("Using features:", features)

# choose target
if "next_day_dir" in df.columns:
    target_col = "next_day_dir"
elif "next_day_return" in df.columns:
    target_col = "next_day_return"
    df[target_col] = (df[target_col] > 0).astype(int)
else:
    raise RuntimeError("No target column found. Expected 'next_day_dir' or 'next_day_return'.")

df = df.dropna(subset=features + [target_col]).reset_index(drop=True)
print("Total rows after dropna:", len(df))

def build_sequences(df, feature_cols, target_col, lookback=14):
    X, y = [], []
    for i in range(lookback, len(df)):
        seq = df.loc[i - lookback:i - 1, feature_cols].values
        X.append(seq)
        y.append(df.loc[i, target_col])
    return np.asarray(X).astype(np.float32), np.asarray(y).astype(np.int32)

X, y = build_sequences(df, features, target_col, LOOKBACK)
print("Sequence dataset shapes X:", X.shape, "y:", y.shape)

# time-aware splits
split_idx = int((1 - TEST_SIZE) * len(X))
X_train_all, X_test = X[:split_idx], X[split_idx:]
y_train_all, y_test = y[:split_idx], y[split_idx:]
print("Train_all / Test shapes:", X_train_all.shape, X_test.shape)

val_count = int(VAL_SIZE * len(X_train_all))
if val_count < 5:
    val_count = max(1, int(0.1 * len(X_train_all)))
train_count = len(X_train_all) - val_count
X_train, X_val = X_train_all[:train_count], X_train_all[train_count:]
y_train, y_val = y_train_all[:train_count], y_train_all[train_count:]
print("Train / Val shapes:", X_train.shape, X_val.shape)

# optional oversampling
if USE_OVERSAMPLING:
    try:
        from imblearn.over_sampling import RandomOverSampler
        ns, ts, nf = X_train.shape
        X_flat = X_train.reshape(ns, ts * nf)
        ros = RandomOverSampler(random_state=SEED)
        X_res, y_res = ros.fit_resample(X_flat, y_train)
        X_train = X_res.reshape(-1, ts, nf)
        y_train = y_res
        print("After oversampling, train shape:", X_train.shape, "class dist:", np.unique(y_train, return_counts=True))
    except Exception as e:
        print("Oversampling requested but failed:", e)
        print("Continuing without oversampling.")

# scaler
nsamples, ntimesteps, nfeat = X_train.shape
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, nfeat)
X_val_flat = X_val.reshape(-1, nfeat)
X_test_flat = X_test.reshape(-1, nfeat)
scaler.fit(X_train_flat)
X_train_scaled = scaler.transform(X_train_flat).reshape(nsamples, ntimesteps, nfeat)
X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape[0], ntimesteps, nfeat)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape[0], ntimesteps, nfeat)

# class weights
from sklearn.utils import class_weight
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: float(w) for i, w in enumerate(cw)}
print("Train class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Using class_weights:", class_weights)

# ------------- model builders -------------
def build_bidirectional_lstm_with_attention(input_shape, units=64, dropout=0.3):
    inp = layers.Input(shape=input_shape)
    x = layers.Masking()(inp)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    att = AttentionLayer()(x)
    x = layers.Dropout(dropout)(att)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

def build_lstm_baseline_simple(input_shape, units=64, dropout=0.3):
    inp = layers.Input(shape=input_shape)
    x = layers.Masking()(inp)
    x = layers.LSTM(units, return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

input_shape = (LOOKBACK, nfeat)
print("Input shape:", input_shape)

loss_fn = focal_loss(alpha=0.6, gamma=2.0) if USE_FOCAL_LOSS else 'binary_crossentropy'

# IMPORTANT: create SEPARATE optimizer instances for each model (fixes KeyError)
optimizer_baseline = tf.keras.optimizers.Adam(learning_rate=1e-4)
optimizer_attn = tf.keras.optimizers.Adam(learning_rate=1e-4)

model_baseline = build_lstm_baseline_simple(input_shape, units=64, dropout=0.3)
model_baseline.compile(optimizer=optimizer_baseline, loss=loss_fn, metrics=['accuracy'])

model_attn = build_bidirectional_lstm_with_attention(input_shape, units=48, dropout=0.35)
model_attn.compile(optimizer=optimizer_attn, loss=loss_fn, metrics=['accuracy'])

print(model_baseline.summary())
print(model_attn.summary())

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-7)
]

# ------------- train -------------
history_baseline = model_baseline.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

history_attn = model_attn.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

# save models
model_baseline.save(os.path.join(MODEL_DIR, "lstm_baseline.keras"))
model_attn.save(os.path.join(MODEL_DIR, "lstm_attention.keras"))
print("Saved models to", MODEL_DIR)

# ------------- threshold tuning & evaluation -------------
from sklearn.metrics import precision_recall_curve
def find_best_threshold(model, X_val, y_val):
    probs = model.predict(X_val, batch_size=1024).ravel()
    prec, rec, thr = precision_recall_curve(y_val, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.nanargmax(f1s)
    if best_idx >= len(thr):
        return 0.5
    return float(thr[best_idx])

best_thresh_baseline = find_best_threshold(model_baseline, X_val_scaled, y_val)
best_thresh_attn = find_best_threshold(model_attn, X_val_scaled, y_val)
print("Best thresholds (val) -> baseline:", best_thresh_baseline, "attn:", best_thresh_attn)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    probs = model.predict(X_test, batch_size=1024).ravel()
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = None
    cm = confusion_matrix(y_test, preds)
    bal = balanced_accuracy_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    return {"probs": probs, "preds": preds, "acc": acc, "prec": prec, "rec": rec,
            "f1": f1, "auc": auc, "cm": cm, "report": report, "bal": bal, "mcc": mcc}

res_baseline = evaluate_model(model_baseline, X_test_scaled, y_test, threshold=best_thresh_baseline)
res_attn = evaluate_model(model_attn, X_test_scaled, y_test, threshold=best_thresh_attn)

def print_eval(name, r):
    print("="*40)
    print("Model:", name)
    print(f"Accuracy: {r['acc']:.4f}  BalancedAcc: {r['bal']:.4f}  MCC: {r['mcc']:.4f}")
    print(f"Precision: {r['prec']:.4f}  Recall: {r['rec']:.4f}  F1: {r['f1']:.4f}")
    print("ROC AUC:", r['auc'])
    print("Confusion Matrix:\n", r['cm'])
    print("Classification Report:\n", r['report'])

print_eval("LSTM Baseline (tuned thresh)", res_baseline)
print_eval("LSTM + Attention (tuned thresh)", res_attn)

# save test predictions
pd.DataFrame({
    "prob_baseline": res_baseline["probs"],
    "pred_baseline": res_baseline["preds"],
    "prob_attn": res_attn["probs"],
    "pred_attn": res_attn["preds"],
    "y_true": y_test
}).to_csv(os.path.join(MODEL_DIR, "test_predictions_tuned.csv"), index=False)
print("Saved test predictions to", os.path.join(MODEL_DIR, "test_predictions_tuned.csv"))

# plots + summary
def plot_history(h1, h2=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(h1.history['loss'], label='train baseline loss')
    plt.plot(h1.history['val_loss'], label='val baseline loss')
    if h2:
        plt.plot(h2.history['loss'], '--', label='train attn loss')
        plt.plot(h2.history['val_loss'], '--', label='val attn loss')
    plt.title("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(h1.history.get('accuracy', h1.history.get('acc')), label='train baseline acc')
    plt.plot(h1.history.get('val_accuracy', h1.history.get('val_acc')), label='val baseline acc')
    if h2:
        plt.plot(h2.history.get('accuracy', h2.history.get('acc')), '--', label='train attn acc')
        plt.plot(h2.history.get('val_accuracy', h2.history.get('val_acc')), '--', label='val attn acc')
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"))
    print("Saved training curves to", os.path.join(MODEL_DIR, "training_curves.png"))

plot_history(history_baseline, history_attn)

summary = pd.DataFrame({
    "model": ["LSTM_baseline", "LSTM_attention"],
    "accuracy": [res_baseline["acc"], res_attn["acc"]],
    "balanced_acc": [res_baseline["bal"], res_attn["bal"]],
    "mcc": [res_baseline["mcc"], res_attn["mcc"]],
    "precision": [res_baseline["prec"], res_attn["prec"]],
    "recall": [res_baseline["rec"], res_attn["rec"]],
    "f1": [res_baseline["f1"], res_attn["f1"]],
    "roc_auc": [res_baseline["auc"], res_attn["auc"]],
})
summary.to_csv(os.path.join(MODEL_DIR, "comparison_metrics_tuned.csv"), index=False)
print("Comparison saved to", os.path.join(MODEL_DIR, "comparison_metrics_tuned.csv"))
print("Comparison metrics:\n", summary)