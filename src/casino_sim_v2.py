# ============================================================
# CASINO SIM LAB V2 - Módulo Estadístico
# ============================================================
import os, json, math, hashlib, hmac
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SEED = 1
OUT = Path("reports")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)
RED = np.array([1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36])

def roulette(n, rng):
    numbers = rng.integers(0, 37, n)
    return np.where(np.isin(numbers, RED), 1.0, -1.0)

def slot_exact():
    total = (8**3 * 5 + 6**3 * 10 + 3**3 * 25 + 100)
    return total / 18**3

def craps_exact():
    from fractions import Fraction
    ev = Fraction(8 - 4, 36)
    for w in (3, 4, 5, 5, 4, 3):
        ev += (Fraction(w, 36) * Fraction(w - 6, w + 6))
    return float(ev)

def session(system, n=30, bankroll=200, cap=64, rng=None):
    bal = float(bankroll)
    stake = 1.0
    stopped = False
    for _ in range(n):
        if bal < stake:
            stopped = True
            break
        win = bool(rng.integers(0, 37) in RED)
        bal += stake * (1 if win else -1)
        if system == "martingala":
            stake = 1 if win or 2 * stake > cap else 2 * stake
        elif system == "dalembert":
            stake = max(1, stake - 1) if win else min(cap, stake + 1)
    return bal - bankroll, stopped

def learn_roulette(train_n=100000, test_n=100000):
    train = roulette(train_n, np.random.default_rng(SEED))
    test = roulette(test_n, np.random.default_rng(SEED + 1))
    checkpoints = np.unique(np.geomspace(100, train_n, 25).astype(int))
    estimates = np.cumsum(train)[checkpoints - 1] / checkpoints
    results = {
        "theoretical_ev": -1 / 37,
        "learned_ev": float(train.mean()),
        "test_ev": float(test.mean()),
        "absolute_error": float(abs(train.mean() - (-1 / 37)))
    }
    return results, checkpoints, estimates

if __name__ == "__main__":
    learned, steps, est = learn_roulette()
    results = {
        "metadata": {"seed": SEED, "note": "Modelo educativo hipotético."},
        "roulette_learning": learned,
        "slot_rtp_exact": slot_exact(),
        "craps_ev_exact": craps_exact()
    }
    
    for system in ("fija", "martingala", "dalembert"):
        local = np.random.default_rng(SEED)
        sessions_res = [session(system, rng=local) for _ in range(3000)]
        profits = np.array([v[0] for v in sessions_res])
        results[system] = {
            "positive_pct": float(100 * np.mean(profits > 0)),
            "unable_to_continue_pct": float(100 * np.mean([v[1] for v in sessions_res])),
            "mean_profit": float(profits.mean()),
            "median_profit": float(np.median(profits))
        }

    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    plt.figure(figsize=(8, 4))
    plt.plot(steps, est, label="Estimación acumulada")
    plt.axhline(-1 / 37, linestyle="--", label="EV teórico")
    plt.xscale("log")
    plt.xlabel("Tiradas de entrenamiento")
    plt.ylabel("Ganancia neta / unidad")
    plt.title("Autoaprendizaje estadístico: ruleta")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "roulette_learning.png", dpi=150)
    plt.close()
    print("Simulación V2 completada y guardada en reports/")
