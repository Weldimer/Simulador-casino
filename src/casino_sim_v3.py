# ============================================================
# CASINO SIM LAB V3 — NEURAL ENGINE
# Electrónica PERLACO
# Red neuronal de aprendizaje supervisado con TensorFlow.
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEED_NN = 42
np.random.seed(SEED_NN)
tf.keras.utils.set_random_seed(SEED_NN)

OUT = Path("reports")
OUT.mkdir(exist_ok=True)
MODELS_OUT = Path("models")
MODELS_OUT.mkdir(exist_ok=True)

MULTS_NN = np.array([
    0.1, 0.2, 0.3, 0.5,
    1, 2, 3, 5,
    10, 20, 50, 100,
    500, 1000
], dtype=float)

def nn_distribution(rtp, freq=0.20):
    def weights(alpha):
        w = MULTS_NN ** (-alpha)
        return freq * w / w.sum()
    lo, hi = -2.0, 8.0
    if not (weights(hi) @ MULTS_NN <= rtp <= weights(lo) @ MULTS_NN):
        raise ValueError("RTP incompatible")
    for _ in range(80):
        mid = (lo + hi) / 2
        if weights(mid) @ MULTS_NN > rtp:
            lo = mid
        else:
            hi = mid
    p = weights((lo + hi) / 2)
    return np.r_[MULTS_NN, 0.0], np.r_[p, 1 - freq]

def generate_dataset(scenarios=400, sessions_per_scenario=64):
    rng = np.random.default_rng(SEED_NN)
    X, Y = [], []
    print("Generando dataset sintético para el modelo neuronal...")
    for i in range(scenarios):
        rtp = rng.uniform(0.85, 0.99)
        freq = 0.20
        spins = int(rng.integers(30, 200))
        bankroll = rng.uniform(10, 100)
        bet = rng.uniform(0.10, 0.50)
        vals, probs = nn_distribution(rtp, freq)
        variance = np.sum(probs * (vals - rtp)**2)
        
        draws = rng.choice(vals, size=(sessions_per_scenario, spins), p=probs)
        balances = bankroll + np.cumsum(bet * (draws - 1), axis=1)
        cannot_continue = np.any(balances < bet - 1e-9, axis=1)
        ruin_probability = float(cannot_continue.mean())

        X.append([rtp, freq, spins, bankroll, bet, np.sqrt(variance)])
        Y.append([rtp, ruin_probability])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)

if __name__ == "__main__":
    X, Y = generate_dataset()
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.30, random_state=SEED_NN)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEED_NN)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    neural_model = Sequential([
        Input(shape=(6,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dropout(0.10),
        Dense(16, activation="relu"),
        Dense(2, activation="sigmoid")
    ])

    neural_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    history = neural_model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    pred = neural_model.predict(X_test_s, verbose=0)
    mae_rtp = float(mean_absolute_error(y_test[:, 0], pred[:, 0]))
    mae_ruin = float(mean_absolute_error(y_test[:, 1], pred[:, 1]))

    # Gráfica de entrenamiento
    plt.figure(figsize=(9, 4))
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.title("Casino Sim Lab V3 - Aprendizaje neuronal")
    plt.xlabel("Épocas")
    plt.ylabel("Error cuadrático medio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "neural_training.png", dpi=150)
    plt.close()

    # Guardar modelo y escalador
    neural_model.save(MODELS_OUT / "casino_neural_v3.keras")
    np.savez(MODELS_OUT / "neural_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

    metrics = {
        "version": "V3",
        "model": "MLP supervised",
        "seed": SEED_NN,
        "mae_rtp": mae_rtp,
        "mae_ruin": mae_ruin
    }
    (OUT / "neural_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Entrenamiento V3 finalizado y artefactos guardados.")
