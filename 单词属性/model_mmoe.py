import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import style_utils

def train_mmoe(df, feature_cols):
    print("\n" + "="*40)
    print(">>> Advanced: MMoE Deep Learning Model")
    print("="*40)
    
    # Prepare Data (Target is the full distribution)
    dist_cols = ['1_try', '2_tries', '3_tries', '4_tries', '5_tries', '6_tries', '7_or_more_tries_x']
    valid_idx = df[dist_cols].dropna().index
    X = df.loc[valid_idx, feature_cols].values
    y = df.loc[valid_idx, dist_cols].values
    
    # Standardization (Crucial for Neural Networks)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- MMoE Model Definition (Fixed for Keras 3 compatibility) ---
    input_dim = X.shape[1]
    input_layer = Input(shape=(input_dim,))
    
    # 1. Experts Layer
    experts = []
    for i in range(4): # 4 Experts
        e = layers.Dense(32, activation='relu')(input_layer)
        e = layers.Dense(16, activation='relu')(e)
        experts.append(e)
    
    # Stack experts outputs: [batch, num_experts, expert_dim]
    # FIX: Use Lambda layer for tf ops
    experts_tensor = layers.Lambda(lambda x: tf.stack(x, axis=1))(experts)
    
    # 2. Multi-Gate & Towers
    towers = []
    for i in range(7): # 7 Tasks (Distribution bins)
        # Gate: weighted sum of experts
        gate = layers.Dense(4, activation='softmax')(input_layer)
        gate = layers.Lambda(lambda x: tf.expand_dims(x, -1))(gate) # [batch, 4, 1]
        
        # Weighted sum: Experts * Gate
        # FIX: Use Lambda for multiplication and reduction
        weighted_expert = layers.Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([experts_tensor, gate])
        
        # Task Tower
        tower = layers.Dense(8, activation='relu')(weighted_expert)
        out = layers.Dense(1)(tower)
        towers.append(out)
        
    # 3. Output Layer (Softmax to ensure sum=1)
    concat = layers.Concatenate()(towers)
    output = layers.Softmax()(concat)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='kl_divergence') # KL Divergence is best for distributions
    
    # Train
    print("Training MMoE...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
    
    # --- Evaluation ---
    y_pred = model.predict(X_test)
    
    # Calculate MSE and MAE
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    
    print("\n[MMoE Model Performance]")
    print(f"  MSE: {mse:.6f} (Target: Distribution)")
    print(f"  MAE: {mae:.6f}")
    print(f"  R2:  {r2:.4f}")
    
    # Plot Learning Curve
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss', color=style_utils.MORANDI_COLORS[0], linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color=style_utils.MORANDI_COLORS[1], linewidth=2)
    plt.title("MMoE Learning Curve (KL Divergence)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()