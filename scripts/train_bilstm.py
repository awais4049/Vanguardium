"""
Vanguardium - BiLSTM Training (CICIDS 2017 network-flow classifier)

Reproducible training script for a BiLSTM model on CICIDS 2017 data.
Features are grouped into 11 semantic categories (packet size, timing,
rate, header/protocol, etc.) and zero-padded to width 10, forming a
pseudo-sequence input of shape (11, 10) — a recognised approach in IDS
literature when true temporal sequences are unavailable.

Usage (from project root, with venv active):
    python scripts/train_bilstm.py

Input:
    data/processed/cicids2017_processed.csv

Output (all to data/models/):
    bilstm_model.keras          - trained BiLSTM model
    bilstm_scaler.pkl           - StandardScaler fitted on training data
    bilstm_label_encoder.pkl    - LabelEncoder (class name ↔ int)
    bilstm_feature_groups.json  - semantic feature grouping definition
    bilstm_results.json         - metrics, class weights, report

Design decision: class-weighted model selected over unweighted despite
near-identical macro F1 because it improves recall on minority attack
classes (Bots: 0.89→0.95, Web Attacks: 0.93→0.97) at the cost of
precision on Normal Traffic, reflecting IDS priorities where missed
attacks (false negatives) cost more than false alarms (false positives).
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

FEATURE_GROUPS = {
    "flow_identity": ["Destination Port", "Flow Duration"],
    "packet_volume": ["Total Fwd Packets", "Total Length of Fwd Packets", "Subflow Fwd Bytes", "act_data_pkt_fwd"],
    "fwd_packet_size": ["Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std"],
    "bwd_packet_size": ["Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std"],
    "overall_packet_size": ["Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "Average Packet Size"],
    "flow_rate": ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s"],
    "flow_timing": ["Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min"],
    "fwd_timing": ["Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min"],
    "bwd_timing": ["Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min"],
    "activity_timing": ["Active Mean", "Active Max", "Active Min", "Idle Mean", "Idle Max", "Idle Min"],
    "header_protocol": ["Fwd Header Length", "Bwd Header Length", "FIN Flag Count", "PSH Flag Count", "ACK Flag Count", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "min_seg_size_forward", "Init_Win_bytes_forward_missing", "Init_Win_bytes_backward_missing"]
}

def create_sequences(df, feature_groups, max_features_per_group=10):
    num_samples = len(df)
    num_timesteps = len(feature_groups)
    
    # Initialize with zeros (padding)
    sequences = np.zeros((num_samples, num_timesteps, max_features_per_group))
    
    for t, (group_name, features) in enumerate(feature_groups.items()):
        # Select available features for this group
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            continue
            
        # Get values for available features
        group_data = df[available_features].values
        
        # Determine how many features to copy (up to max_features_per_group)
        n_features = min(len(available_features), max_features_per_group)
        
        # Place the data into the sequences array
        sequences[:, t, :n_features] = group_data[:, :n_features]
        
    return sequences

def main():
    print("Loading data...")
    data_path = os.path.join('data', 'processed', 'cicids2017_processed.csv')
    df = pd.read_csv(data_path)
    
    # Drop NaNs if any
    df = df.dropna()
    
    print("Extracting features and labels...")
    label_col = 'Attack Type'
    y = df[label_col].values
    
    # Identify all features used in groups
    all_features = []
    for feats in FEATURE_GROUPS.values():
        all_features.extend(feats)
        
    # Keep only available features
    available_features = [f for f in all_features if f in df.columns]
    X_df = df[available_features]
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_df)
    
    # Reconstruct scaled dataframe
    X_scaled_df = pd.DataFrame(X_scaled_array, columns=available_features)
    
    print("Grouping features into sequences...")
    X = create_sequences(X_scaled_df, FEATURE_GROUPS, max_features_per_group=10)
    
    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    
    print("Computing class weights...")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {c: w for c, w in zip(classes, weights)}
    print(f"Class weights: {class_weight_dict}")
    
    print("Building model...")
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(11, 10)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    print("Training model...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print("Evaluating model...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    class_names = [str(c) for c in le.classes_]
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("Saving artifacts...")
    models_dir = os.path.join('data', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model.save(os.path.join(models_dir, 'bilstm_model.keras'))
    joblib.dump(scaler, os.path.join(models_dir, 'bilstm_scaler.pkl'))
    joblib.dump(le, os.path.join(models_dir, 'bilstm_label_encoder.pkl'))
    
    with open(os.path.join(models_dir, 'bilstm_feature_groups.json'), 'w') as f:
        json.dump(FEATURE_GROUPS, f, indent=4)
        
    results = {
        'model': 'BiLSTM (2-layer bidirectional, class-weighted)',
        'architecture': 'BiLSTM(64, return_sequences=True) -> Dropout(0.3) -> BiLSTM(32) -> Dropout(0.3) -> Dense(32, relu) -> Dense(7, softmax)',
        'input_shape': '(11 timesteps, 10 features) - pseudo-sequence via semantic feature grouping',
        'test_accuracy': acc,
        'test_macro_f1': macro_f1,
        'test_weighted_f1': weighted_f1,
        'classification_report': report,
        'class_weights': {le.classes_[k]: float(v) for k, v in class_weight_dict.items()},
    }
    with open(os.path.join(models_dir, 'bilstm_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Training complete! Artifacts saved to data/models/")

if __name__ == '__main__':
    main()
